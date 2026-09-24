#!/usr/bin/env python3
"""Greedy closed-loop cube evaluation, without search or action filters."""
import argparse
import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path

from cube_smoke import ACTIONS, action_logits

SOLVED = ''.join(f * 9 for f in 'URFDLB')
# Exact facelet permutations derived from cubejs 1.3.2 cp/co/ep/eo mappings.
# Pure Python execution on the training server needs no Node.js or cube package.
QUARTER_TURNS = {
    'U': [6, 3, 0, 7, 4, 1, 8, 5, 2, 45, 46, 47, 12, 13, 14, 15, 16, 17, 9, 10, 11, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 18, 19, 20, 39, 40, 41, 42, 43, 44, 36, 37, 38, 48, 49, 50, 51, 52, 53],
    'R': [0, 1, 20, 3, 4, 23, 6, 7, 26, 15, 12, 9, 16, 13, 10, 17, 14, 11, 18, 19, 29, 21, 22, 32, 24, 25, 35, 27, 28, 51, 30, 31, 48, 33, 34, 45, 36, 37, 38, 39, 40, 41, 42, 43, 44, 8, 46, 47, 5, 49, 50, 2, 52, 53],
    'F': [0, 1, 2, 3, 4, 5, 44, 41, 38, 6, 10, 11, 7, 13, 14, 8, 16, 17, 24, 21, 18, 25, 22, 19, 26, 23, 20, 15, 12, 9, 30, 31, 32, 33, 34, 35, 36, 37, 27, 39, 40, 28, 42, 43, 29, 45, 46, 47, 48, 49, 50, 51, 52, 53],
    'D': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 24, 25, 26, 18, 19, 20, 21, 22, 23, 42, 43, 44, 33, 30, 27, 34, 31, 28, 35, 32, 29, 36, 37, 38, 39, 40, 41, 51, 52, 53, 45, 46, 47, 48, 49, 50, 15, 16, 17],
    'L': [53, 1, 2, 50, 4, 5, 47, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 19, 20, 3, 22, 23, 6, 25, 26, 18, 28, 29, 21, 31, 32, 24, 34, 35, 42, 39, 36, 43, 40, 37, 44, 41, 38, 45, 46, 33, 48, 49, 30, 51, 52, 27],
    'B': [11, 14, 17, 3, 4, 5, 6, 7, 8, 9, 10, 35, 12, 13, 34, 15, 16, 33, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 36, 39, 42, 2, 37, 38, 1, 40, 41, 0, 43, 44, 51, 48, 45, 52, 49, 46, 53, 50, 47],
}


def move(state, action):
    count = 2 if action.endswith('2') else 3 if action.endswith("'") else 1
    for _ in range(count):
        state = ''.join(state[i] for i in QUARTER_TURNS[action[0]])
    return state


def distances_to_goal():
    distances = {SOLVED: 0}
    frontier = [SOLVED]
    for depth in range(1, 4):
        following = []
        for state in frontier:
            for action in ACTIONS:
                child = move(state, action)
                if child not in distances:
                    distances[child] = depth
                    following.append(child)
        frontier = following
    return distances


def make_cases(rows, seed):
    # Solver distances are used ONLY for case construction and reporting.
    distances = distances_to_goal()
    training = set()
    cases = []
    for row in rows:
        state = row['state']
        depth = distances.get(state)
        if state in training or depth not in (1, 2, 3) or depth != row['optimal_distance']:
            raise ValueError('Invalid/duplicate training state or inconsistent depth')
        training.add(state)
        cases.append({'id': row['id'], 'state': state, 'depth': depth, 'split': 'train'})
    rng = random.Random(seed)
    for depth in (2, 3):
        available = sorted(s for s, d in distances.items() if d == depth and s not in training)
        chosen = rng.sample(available, min(256, len(available)))
        cases.extend({'id': f'heldout-{depth}-{i:04d}', 'state': s, 'depth': depth, 'split': 'heldout'}
                     for i, s in enumerate(chosen))
    return cases


def rollout(cases, predict, max_steps):
    results = [{**c, 'initial_state': c['state'], 'moves': [], 'solved': c['state'] == SOLVED,
                'repeated_states': 0} for c in cases]
    seen = [{c['state']} for c in cases]
    for _ in range(max_steps):
        active = [i for i, r in enumerate(results) if not r['solved']]
        if not active:
            break
        chosen = predict([results[i]['state'] for i in active])
        if len(chosen) != len(active):
            raise ValueError('Prediction count mismatch')
        for i, index in zip(active, chosen):
            if type(index) is not int or not 0 <= index < 18:
                raise ValueError('Invalid predicted action index')
            r = results[i]
            action = ACTIONS[index]
            r['state'] = move(r['state'], action)
            r['moves'].append(action)
            r['repeated_states'] += int(r['state'] in seen[i])
            seen[i].add(r['state'])
            r['solved'] = r['state'] == SOLVED
    return results


def summarize(results):
    successes = [r for r in results if r['solved']]
    return {'cases': len(results), 'solved': len(successes),
            'success_rate': len(successes) / len(results),
            'mean_steps_success': sum(len(r['moves']) for r in successes) / len(successes) if successes else None,
            'shortest_solve_rate': sum(r['solved'] and len(r['moves']) == r['depth'] for r in results) / len(results),
            'repeat_episode_rate': sum(r['repeated_states'] > 0 for r in results) / len(results)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path, help='Default: exact training_data.jsonl saved beside checkpoint')
    p.add_argument('--output', type=Path)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--max-steps', type=int, default=10)
    p.add_argument('--seed', type=int, default=17)
    args = p.parse_args()
    if args.batch_size < 1 or args.max_steps < 1:
        p.error('Batch size and max steps must be positive')
    data_path = args.data or args.checkpoint.parent / 'training_data.jsonl'
    rows = [json.loads(line) for line in data_path.read_text().splitlines() if line.strip()]
    if not rows:
        p.error('Training data cannot be empty')
    saved_actions = json.loads((args.checkpoint / 'actions.json').read_text())
    if saved_actions != list(ACTIONS):
        p.error('Checkpoint action ordering mismatch')
    cases = make_cases(rows, args.seed)
    output = args.output or args.checkpoint.parent / f'eval-{datetime.now():%Y%m%d-%H%M%S}'
    output.mkdir(parents=True, exist_ok=False)
    (output / 'cases.jsonl').write_text(''.join(json.dumps(c) + '\n' for c in cases))
    print('Cases:', dict(Counter(f"{c['split']}_depth{c['depth']}" for c in cases)), flush=True)
    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen3_5TextModel, Qwen3_5TextConfig
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA GPU with BF16 support required')
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint / 'tokenizer', local_files_only=True, padding_side='right')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Explicit text config avoids treating the saved text-only body as a full VL model.
    cfg = Qwen3_5TextConfig(**json.loads((args.checkpoint / 'backbone/config.json').read_text()))
    body, info = Qwen3_5TextModel.from_pretrained(args.checkpoint / 'backbone', config=cfg,
        local_files_only=True, dtype=torch.float32, attn_implementation='sdpa', output_loading_info=True)
    if any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise RuntimeError(f'Checkpoint load mismatch: {info}')
    body = body.to('cuda').eval()
    head = torch.nn.Linear(body.config.hidden_size, 18)
    head.load_state_dict(torch.load(args.checkpoint / 'action_head.pt', map_location='cpu', weights_only=True))
    head = head.to('cuda').eval()

    @torch.inference_mode()
    def predict(states):
        choices = []
        for start in range(0, len(states), args.batch_size):
            texts = ['\n'.join(f'{f}: {s[i*9:i*9+9]}' for i, f in enumerate('URFDLB')) + '\nAction:'
                     for s in states[start:start + args.batch_size]]
            inputs = tokenizer(texts, padding=True, truncation=False, return_tensors='pt', return_token_type_ids=False)
            if inputs['input_ids'].shape[1] > min(512, body.config.max_position_embeddings):
                raise ValueError('Refusing to truncate input')
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = action_logits(body, head, inputs)
            if not torch.isfinite(logits).all():
                raise RuntimeError('Nonfinite action scores')
            choices.extend(logits.argmax(-1).cpu().tolist())
        return choices

    results = []
    with (output / 'episodes.jsonl').open('w') as log:
        for start in range(0, len(cases), args.batch_size):
            batch = rollout(cases[start:start + args.batch_size], predict, args.max_steps)
            # Replay all recorded moves to verify final state and solved flag.
            for r in batch:
                replay = r['initial_state']
                for a in r['moves']:
                    replay = move(replay, a)
                if replay != r['state'] or (replay == SOLVED) != r['solved']:
                    raise RuntimeError('Episode replay mismatch')
                log.write(json.dumps(r) + '\n')
            log.flush()
            results.extend(batch)
            print(f"Evaluated {len(results)}/{len(cases)}", flush=True)
    summary = {'checkpoint': str(args.checkpoint.resolve()), 'data': str(data_path.resolve()),
               'max_steps': args.max_steps, 'seed': args.seed, 'controller': 'greedy argmax, no action filters or search',
               'heldout_scope': 'Initial states excluded from training; subsequent states may overlap training.',
               'torch': torch.__version__, 'transformers': transformers.__version__, 'metrics': {}}
    for split in ('train', 'heldout'):
        subset = [r for r in results if r['split'] == split]
        if subset:
            summary['metrics'][split] = summarize(subset)
        for depth in (1, 2, 3):
            group = [r for r in subset if r['depth'] == depth]
            if group:
                summary['metrics'][f'{split}_depth{depth}'] = summarize(group)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
    print(f'Saved evaluation: {output}', flush=True)


if __name__ == '__main__':
    main()
