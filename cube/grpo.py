#!/usr/bin/env python3
"""Grouped online cube RL + differentiable Brier confidence loss.
Not a reproduction of Jev RLCD or the RLCR paper's generated-confidence policy.
Actor input/18-action head stay identical to SFT; confidence also sees remaining budget.
"""
import argparse
import copy
import json
import math
import random
from pathlib import Path
import time
import torch
from cube.eval import ACTIONS, SOLVED, move, distances_to_goal, make_cases
from cube.train import read_data, save_model


def group_advantages(outcomes):
    rewards = torch.tensor(outcomes, dtype=torch.float32)
    return (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-8)


@torch.no_grad()
def collect_groups(policy, starts, group_size, max_steps, forbidden, temperature=1.0, return_all=False):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    episodes = [{'start': s, 'state': s, 'steps': [], 'outcome': 0, 'invalid': False}
                for s in starts for _ in range(group_size)]
    for step in range(max_steps):
        active = []
        for i, e in enumerate(episodes):
            if e['state'] in forbidden:
                e['invalid'] = True
            if not e['invalid'] and not e['outcome']:
                active.append(i)
        if not active:
            break
        logits, confidence = policy([episodes[i]['state'] for i in active], [max_steps-step]*len(active))
        if not torch.isfinite(logits).all() or not torch.isfinite(confidence).all():
            raise RuntimeError('Nonfinite rollout predictions')
        logp = (logits.float()/temperature).log_softmax(-1)
        entropy = -(logp.exp()*logp).sum(-1)
        selected = torch.multinomial(logp.exp(), 1).squeeze(-1)
        for j, i in enumerate(active):
            e = episodes[i]
            a = int(selected[j])
            e['steps'].append({'state': e['state'], 'budget': max_steps-step, 'action': ACTIONS[a],
                'index': a, 'old_logp': float(logp[j, a]), 'confidence': float(confidence[j]),
                'entropy_nats': float(entropy[j]), 'top1_probability': float(logp[j].max().exp())})
            e['state'] = move(e['state'], ACTIONS[a])
            e['invalid'] |= e['state'] in forbidden
            e['outcome'] = int(e['state'] == SOLVED)
    # Drop the ENTIRE group if any trajectory reaches a reserved test start.
    # This avoids biased within-group baselines from dropping individual failures.
    groups = []
    for start in range(0, len(episodes), group_size):
        group = episodes[start:start+group_size]
        if return_all or not any(e['invalid'] for e in group):
            groups.append(group)
    return groups


def objective(logits, confidence, actions, old_logp, reference_logp, advantages,
              outcomes, weights, clip, kl_weight, calibration_weight):
    logp = logits.float().log_softmax(-1)
    selected = logp.gather(1, actions[:, None]).squeeze(1)
    ratio = (selected-old_logp.detach()).exp()
    advantage = advantages.detach()
    actor = -torch.minimum(ratio*advantage, ratio.clamp(1-clip, 1+clip)*advantage)
    kl = (logp.exp() * (logp-reference_logp.detach())).sum(-1)
    brier = (confidence-outcomes.detach()).square()
    loss = (weights*(actor + kl_weight*kl + calibration_weight*brier)).sum()
    return loss, {'policy_loss': float((weights*actor).sum().detach()),
                  'kl': float((weights*kl).sum().detach()),
                  'brier': float((weights*brier).sum().detach())}


class Agent(torch.nn.Module):
    def __init__(self, body, head, tokenizer, max_steps, device):
        super().__init__()
        self.body, self.head, self.tokenizer = body, head, tokenizer
        self.max_steps, self.device_name = max_steps, device
        self.confidence = torch.nn.Linear(body.config.hidden_size+1, 1).to(device)
        torch.nn.init.zeros_(self.confidence.weight)
        torch.nn.init.zeros_(self.confidence.bias)

    def forward(self, states, budgets):
        texts = ['\n'.join(f'{f}: {s[i*9:i*9+9]}' for i,f in enumerate('URFDLB'))+'\nAction:' for s in states]
        tokens = self.tokenizer(texts, padding=True, truncation=False, return_tensors='pt', return_token_type_ids=False)
        if tokens['input_ids'].shape[1] > min(512, self.body.config.max_position_embeddings):
            raise ValueError('Refusing to truncate input')
        tokens = {k:v.to(self.device_name) for k,v in tokens.items()}
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.device_name == 'cuda'):
            hidden = self.body(**tokens, use_cache=False).last_hidden_state
            last = tokens['attention_mask'].sum(-1)-1
            h = hidden[torch.arange(len(states), device=self.device_name), last].float()
            logits = self.head(h).float()
        # Full precision scalar readout with explicit budget; does not change actor prompt.
        b = torch.tensor(budgets, device=self.device_name, dtype=torch.float32)[:, None]/self.max_steps
        confidence = self.confidence(torch.cat([h, b], -1)).squeeze(-1).sigmoid()
        return logits, confidence


def load_agent(checkpoint, max_steps, device):
    from transformers import AutoTokenizer, Qwen3_5TextConfig, Qwen3_5TextModel
    if json.loads((checkpoint/'actions.json').read_text()) != list(ACTIONS):
        raise ValueError('Action mapping mismatch')
    config = Qwen3_5TextConfig(**json.loads((checkpoint/'backbone/config.json').read_text()))
    body, info = Qwen3_5TextModel.from_pretrained(checkpoint/'backbone', config=config,
        local_files_only=True, dtype=torch.float32, attn_implementation='sdpa', output_loading_info=True)
    if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise ValueError(f'Checkpoint mismatch: {info}')
    head = torch.nn.Linear(config.hidden_size, 18)
    head.load_state_dict(torch.load(checkpoint/'action_head.pt', map_location='cpu', weights_only=True))
    tokenizer = AutoTokenizer.from_pretrained(checkpoint/'tokenizer', local_files_only=True, padding_side='right')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return Agent(body.to(device), head.to(device), tokenizer, max_steps, device)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--eval-data', type=Path, default=Path('data/cube_trajectories_expanded.jsonl'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--iterations', type=int, default=20)
    p.add_argument('--groups', type=int, default=4)
    p.add_argument('--group-size', type=int, default=8)
    p.add_argument('--max-steps', type=int, default=10)
    p.add_argument('--update-epochs', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-6)
    p.add_argument('--head-lr', type=float, default=1e-5)
    p.add_argument('--confidence-lr', type=float, default=1e-4)
    p.add_argument('--kl-weight', type=float, default=0.02)
    p.add_argument('--calibration-weight', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=17)
    args = p.parse_args()
    if min(args.iterations,args.groups,args.max_steps,args.update_epochs) < 1 or args.group_size < 2:
        p.error('Positive counts and group-size >= 2 required')
    if any(not math.isfinite(x) or x <= 0 for x in (args.lr,args.head_lr,args.confidence_lr)):
        p.error('Learning rates must be finite and positive')
    if any(not math.isfinite(x) or x < 0 for x in (args.kl_weight,args.calibration_weight)):
        p.error('Loss weights must be finite and nonnegative')
    if args.output.exists():
        p.error('Choose a new output directory')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        p.error('CUDA BF16 GPU required')
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    training = read_data(args.checkpoint.parent/'training_data.jsonl')
    evaluation_rows = read_data(args.eval_data)
    reserved = {c['state'] for c in make_cases(evaluation_rows, 17) if c['split']=='heldout'}
    if reserved & {r['state'] for r in training}:
        raise ValueError('SFT training data overlaps reserved test starts')
    distances = distances_to_goal()
    # No solver labels enter RL: enumerate legal training STARTS only.
    starts = sorted(s for s,d in distances.items() if d in (2,3) and s not in reserved)
    if args.groups > len(starts):
        p.error('Too many groups for start pool')
    args.output.mkdir(parents=True, exist_ok=False)
    config = {**vars(args), 'method':'GRPO terminal success + differentiable Brier auxiliary head; not original RLCR',
        'start_pool_count':len(starts),'reserved_count':len(reserved),'reward':'1 solved by budget, else 0',
        'temperature':1.0,'reference':'frozen SFT checkpoint, exact categorical KL',
        'confidence_semantics':'success within remaining budget under current temperature-1 sampling policy',
        'confidence_limitation':'Groups visiting reserved starts are discarded; training Brier is selected-data fit, not a calibration evaluation.',
        'input':'actor unchanged state-only; confidence concatenates hidden state and remaining/max_steps',
        'scope':'Single-GPU pilot; no claim of greedy-policy calibration or RLCD reproduction'}
    (args.output/'run.json').write_text(json.dumps(config, default=str, indent=2)+'\n')
    (args.output/'reserved_states.json').write_text(json.dumps(sorted(reserved)))
    (args.output/'rl_start_states.json').write_text(json.dumps(starts))
    (args.output/'training_data.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in training))
    policy = load_agent(args.checkpoint,args.max_steps,'cuda')
    # Exact, fixed SFT reference; 0.8B fits one H800 with both copies.
    reference = copy.deepcopy(policy).eval().requires_grad_(False)
    for module in policy.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    policy.body.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    optimizer = torch.optim.AdamW([
        {'params':policy.body.parameters(),'lr':args.lr},
        {'params':policy.head.parameters(),'lr':args.head_lr},
        {'params':policy.confidence.parameters(),'lr':args.confidence_lr}], weight_decay=0.0)
    started = time.monotonic()
    informative_total = 0
    with (args.output/'metrics.jsonl').open('w') as metrics_file, (args.output/'episodes.jsonl').open('w') as trace:
        for iteration in range(1,args.iterations+1):
            policy.eval()
            groups = collect_groups(policy,rng.sample(starts,args.groups),args.group_size,args.max_steps,reserved)
            episodes = []
            informative = 0
            for group in groups:
                advantages = group_advantages([e['outcome'] for e in group])
                informative += int(bool(advantages.abs().sum()))
                for e,a in zip(group,advantages.tolist()):
                    e['advantage'] = a
                    trace.write(json.dumps({'iteration':iteration,**e})+'\n')
                    episodes.append(e)
            trace.flush()
            informative_total += informative
            stats_total = {}
            optimizer_updates = 0
            # Frozen reference probabilities are scored once per newly collected episode.
            for e in episodes:
                with torch.no_grad():
                    z,_ = reference([r['state'] for r in e['steps']], [r['budget'] for r in e['steps']])
                    e['reference_logp'] = z.log_softmax(-1).detach()
            policy.train()
            for _ in range(args.update_epochs):
                rng.shuffle(episodes)
                # One group-size minibatch, microbatched by episode to bound activation memory.
                for offset in range(0,len(episodes),args.group_size):
                    batch = episodes[offset:offset+args.group_size]
                    optimizer.zero_grad(set_to_none=True)
                    for e in batch:
                        steps = e['steps']; n = len(steps)
                        logits,q = policy([r['state'] for r in steps],[r['budget'] for r in steps])
                        def tensor(values):
                            return torch.tensor(values,device='cuda')
                        loss,stats = objective(logits,q,tensor([r['index'] for r in steps]),
                            tensor([r['old_logp'] for r in steps]),e['reference_logp'],
                            tensor([e['advantage']]*n),tensor([float(e['outcome'])]*n),
                            torch.full((n,),1/(n*len(batch)),device='cuda'),0.2,args.kl_weight,args.calibration_weight)
                        if not torch.isfinite(loss):
                            raise RuntimeError('Nonfinite RL loss')
                        loss.backward()
                        for key, value in stats.items():
                            stats_total[key] = stats_total.get(key, 0.0) + value
                    torch.nn.utils.clip_grad_norm_(policy.parameters(),1.0,error_if_nonfinite=True)
                    optimizer.step()
                    optimizer_updates += 1
            outcomes = [e['outcome'] for e in episodes]
            record = {'iteration':iteration,'accepted_groups':len(groups),'discarded_groups':args.groups-len(groups),
                'informative_groups':informative,'episodes':len(episodes),
                'rollout_success_rate':sum(outcomes)/len(outcomes) if outcomes else None,
                'start_brier_before_update':sum((e['steps'][0]['confidence']-e['outcome'])**2 for e in episodes)/len(episodes) if episodes else None,
                'elapsed_seconds':round(time.monotonic()-started,2),
                'peak_vram_gib':round(torch.cuda.max_memory_allocated()/2**30,2),'optimizer_updates':optimizer_updates,
                **{k:v/optimizer_updates for k,v in stats_total.items()} }
            print(json.dumps(record),flush=True)
            metrics_file.write(json.dumps(record)+'\n'); metrics_file.flush()
    policy.eval()
    if informative_total == 0:
        print('WARNING: No mixed-outcome groups: no reward-driven actor learning occurred; only KL/Brier updates were possible.', flush=True)
    save_model(policy.body,policy.head,args.output/'checkpoint')
    policy.tokenizer.save_pretrained(args.output/'checkpoint/tokenizer')
    torch.save({k:v.detach().cpu() for k,v in policy.confidence.state_dict().items()},args.output/'checkpoint/confidence_head.pt')
    (args.output/'checkpoint/confidence.json').write_text(json.dumps({'input':'concat(last_hidden_float32, remaining/max_steps)',
        'max_steps':args.max_steps,'activation':'sigmoid','policy':'categorical sampling at temperature 1.0',
        'calibration_status':'not independently evaluated; not a calibrated-probability guarantee'},indent=2))
    print(f'Saved {args.output}/checkpoint. Evaluate greedy success separately with the unchanged 332-case panel.',flush=True)


if __name__=='__main__':
    main()
