"""CPU tests for grouped rollout, losses and confidence gradients."""
import torch
from cube.eval import SOLVED, move
from cube.grpo import Agent, group_advantages, collect_groups, objective


def main():
    torch.manual_seed(17)
    assert torch.equal(group_advantages([1, 1, 1]), torch.zeros(3))
    assert torch.equal(group_advantages([0, 0, 0]), torch.zeros(3))
    adv = group_advantages([1, 0, 1, 0])
    assert adv[0] > 0 and adv[1] < 0 and abs(float(adv.mean())) < 1e-6
    class Policy:
        def __call__(self, states, budgets):
            logits = torch.full((len(states), 18), -40.)
            logits[:, 0:2] = 0
            return logits, torch.full((len(states),), 0.5)
    start = move(SOLVED, 'U')
    groups = collect_groups(Policy(), [start], 16, 1, set())
    assert len(groups) == 1 and len(groups[0]) == 16
    assert {e['outcome'] for e in groups[0]} == {0, 1}
    for e in groups[0]:
        assert e['outcome'] == int(move(start, e['steps'][0]['action']) == SOLVED)
    assert collect_groups(Policy(), [start], 4, 2, {start}) == []
    from cube.probe import summarize
    report = summarize(groups)
    assert report['informative_groups'] == 1 and report['all_failure_groups'] == 0
    assert report['mean_unique_first_actions'] == 2
    rejected = collect_groups(Policy(), [start], 4, 2, {start}, return_all=True)
    assert summarize(rejected)['discarded_groups'] == 1
    class PeakedPolicy:
        def __call__(self, states, budgets):
            z = torch.zeros(len(states),18)
            z[:,1] = 6
            return z, torch.full((len(states),),0.5)
    cold = collect_groups(PeakedPolicy(),[start],16,1,set(),temperature=1.)
    warm = collect_groups(PeakedPolicy(),[start],16,1,set(),temperature=1.6)
    assert summarize(warm)['start_entropy_nats'] > summarize(cold)['start_entropy_nats']
    assert summarize(warm)['start_top1_probability'] < summarize(cold)['start_top1_probability']
    z,_ = PeakedPolicy()([start],[1])
    expected = (z/1.6).log_softmax(-1)[0]
    for e in warm[0]:
        row = e['steps'][0]
        assert abs(row['old_logp'] - float(expected[row['index']])) < 1e-6
    for bad in (0., -1., float('nan'), float('inf')):
        try:
            collect_groups(Policy(),[start],4,1,set(),temperature=bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid temperature accepted')
    # A reserved final transition must reject the group too.
    assert collect_groups(PeakedPolicy(),[start],16,1,{SOLVED}) == []
    print('PASS: probe statistics, temperature entropy/log-probabilities, reserved transitions')
    # Correct signs, detached old policy, and an explicit gradient into confidence.
    logits = torch.zeros(2, 18, requires_grad=True)
    qlogits = torch.zeros(2, requires_grad=True)
    old = logits.detach().log_softmax(-1)
    loss, stats = objective(logits, qlogits.sigmoid(), torch.tensor([0, 0]), old[:, 0],
        old, torch.tensor([1., -1.]), torch.tensor([1., 0.]), torch.ones(2)/2, 0.2, 0.02, 0.5)
    loss.backward()
    assert logits.grad[0, 0] < 0 and logits.grad[1, 0] > 0
    assert qlogits.grad[0] < 0 and qlogits.grad[1] > 0
    assert abs(stats['kl']) < 1e-6
    assert all(torch.isfinite(x).all() for x in (logits.grad, qlogits.grad))
    # A deterministic confidence at a shared start only rescales binary rewards;
    # group standardization cancels that scale. It cannot train confidence itself.
    y = torch.tensor([1., 0., 1., 0.])
    r = y - (0.7-y).square()
    torch.testing.assert_close(group_advantages(r.tolist()), group_advantages(y.tolist()))
    from transformers import Qwen3_5TextConfig, Qwen3_5TextModel
    cfg = Qwen3_5TextConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=8, linear_value_head_dim=8,
        layer_types=['linear_attention', 'full_attention'], pad_token_id=0,
        rope_parameters={'rope_type':'default','rope_theta':10000,
                         'partial_rotary_factor':1.0,'mrope_section':[2,3,3]})
    class Tokenizer:
        def __call__(self, texts, **kwargs):
            ids = torch.tensor([[ord(c) % 24 + 1 for c in t[:12]] for t in texts])
            return {'input_ids':ids, 'attention_mask':torch.ones_like(ids)}
    agent = Agent(Qwen3_5TextModel(cfg), torch.nn.Linear(32,18), Tokenizer(), 10, 'cpu')
    agent.body.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    agent.train()
    states, budgets = [start, move(SOLVED,'R')], [10, 9]
    z, q = agent(states,budgets)
    old = z.detach().log_softmax(-1)
    before = agent.head.weight.detach().clone()
    opt = torch.optim.AdamW(agent.parameters(),lr=1e-3)
    loss,_ = objective(z,q,torch.tensor([1,4]),old[[0,1],[1,4]],old,
        torch.tensor([1.,-1.]),torch.tensor([1.,0.]),torch.ones(2)/2,0.2,0.02,0.5)
    loss.backward()
    assert agent.confidence.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.body.parameters())
    opt.step()
    assert not torch.equal(before,agent.head.weight)
    print('PASS: actual hybrid Qwen3.5 GRPO/Brier backward with gradient checkpointing')
    print('PASS: terminal rewards, grouped advantages, no-signal groups, reserved-state rejection, policy/confidence gradients')


if __name__ == '__main__':
    main()
