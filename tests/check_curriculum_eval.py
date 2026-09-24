"""CPU-only checks of curriculum evaluation cases and metric semantics."""
import json
from pathlib import Path
from cube.eval import ACTIONS, SOLVED, make_curriculum_cases, move, rollout, summarize

root=Path(__file__).resolve().parents[1]
def read(name):
    return [json.loads(s) for s in (root/name).read_text().splitlines()]
rows=read('data/curriculum_v1/validation.jsonl')
training=read('data/curriculum_v1/train_upto_5.jsonl')+read('data/cube_trajectories_1000.jsonl')
cases=make_curriculum_cases(rows,training,[4,5])
assert len(cases)==171
assert sum(c['expert_remaining']==4 for c in cases)==77
assert sum(c['expert_remaining']==5 for c in cases)==94
assert cases==make_curriculum_cases(rows,read('data/cube_trajectories_1000.jsonl'),[4,5])
assert all('solution' not in c and 'depth' not in c for c in cases)
# Replay each teacher with a per-case oracle to verify budget and metrics.
by_state={r['state']:r for r in rows}
results=[]
for case in cases:
    sequence=iter(by_state[case['state']]['solution'])
    results+=rollout([case],lambda states:[ACTIONS.index(next(sequence))],case['expert_remaining'])
report=summarize(results)
assert report['success_rate']==report['solve_within_expert_length_rate']==1
assert 'shortest_solve_rate' not in report
for selected, trained, lengths in [(rows,training+[{'state':cases[0]['state']}],[4,5]),
                                  (rows,training,[99]),(rows+[rows[0]],training,[4,5]),
                                  ([{**rows[0], 'solution':['U'] * rows[0]['expert_remaining']}],training,[4,5])]:
    try: make_curriculum_cases(selected,trained,lengths)
    except ValueError: pass
    else: raise AssertionError('Invalid/overlapping/empty selection accepted')
start=move(SOLVED,'R')
case={'id':'budget','state':start,'expert_remaining':1,'split':'validation'}
sequence=iter(['U',"U'","R'"])
r=rollout([case],lambda states:[ACTIONS.index(next(sequence))],3)
assert summarize(r)['success_rate']==1 and summarize(r)['solve_within_expert_length_rate']==0
assert summarize(r)['repeat_episode_rate']==1
print('PASS: identical 171 validation starts, expert rollouts, budget/repeat metrics and overlap rejection')
