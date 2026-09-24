"""Check expert suffixes, split exclusions and rejection of broken trajectories."""
import copy
from pathlib import Path
from cube.curriculum import build, read_rows
from cube.eval import SOLVED, move, distances_to_goal, make_cases

root=Path(__file__).resolve().parents[1]
source=read_rows(root/'data/cube_random_1000/all.jsonl')
previous=read_rows(root/'data/cube_trajectories_1000.jsonl')
evaluation=read_rows(root/'data/cube_trajectories_expanded.jsonl')
samples,report,manifest=build(source,previous,evaluation,20260924)
assert [len(manifest[k]) for k in ('train','validation','test')]==[800,100,100]
assert len(set(manifest['train']+manifest['validation']+manifest['test']))==1000
reserved={r['state'] for r in make_cases(evaluation,17) if r['split']=='heldout'}
trained={r['state'] for r in previous+samples['train']}
assert not reserved & {r['state'] for r in samples['train']}
assert not trained & {r['state'] for r in samples['validation']+samples['test']}
val_states={s for r in source if r['id'] in manifest['validation'] for s in r['states'][:-1]}
assert not val_states & {r['state'] for r in samples['test']}
distances=distances_to_goal()
for name,rows in samples.items():
    assert len(rows)==len({r['state'] for r in rows})
    assert rows
    for row in rows:
        state=row['state']
        assert state!=SOLVED
        assert row['source_id'] in manifest[name]
        assert row['expert_remaining']==len(row['solution'])
        assert row['optimal_distance']==distances.get(state)
        for action in row['solution']: state=move(state,action)
        assert state==SOLVED
    stage=root/f'data/curriculum_v1/{name}.jsonl'
    if stage.exists(): assert read_rows(stage)==rows
for depth in (3,5):
    stage=root/f'data/curriculum_v1/train_upto_{depth}.jsonl'
    if stage.exists():
        assert read_rows(stage)==[r for r in samples['train'] if r['expert_remaining']<=depth]
for broken in ('transition','duplicate'):
    bad=copy.deepcopy(source[:10])
    if broken=='transition': bad[0]['states'][1]=SOLVED
    else: bad[1]=copy.deepcopy(bad[0])
    try: build(bad,previous,evaluation,17)
    except ValueError: pass
    else: raise AssertionError(f'Accepted {broken}')
print('PASS: all curriculum suffixes solve; source splits, exclusions, metadata and broken-input rejection')
print(report['unique_samples'])
