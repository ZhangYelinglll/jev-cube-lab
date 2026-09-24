"""Prepare expert-suffix curriculum data; does not train a model."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
from cube.eval import ACTIONS, SOLVED, distances_to_goal, make_cases, move


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def expand(trajectories, distances):
    samples = {}
    for trajectory in trajectories:
        for step, state in enumerate(trajectory['states'][:-1]):
            suffix = trajectory['solution'][step:]
            candidate = {
                'state': state, 'expert_remaining': len(suffix),
                'optimal_distance': distances.get(state),
                'solution': suffix, 'target_action': suffix[0],
                'target_index': ACTIONS.index(suffix[0]),
                'source_id': trajectory['id'], 'source_step': step,
            }
            # Shortest observed expert suffix, not a claim of optimality.
            if state not in samples or len(suffix) < samples[state]['expert_remaining']:
                samples[state] = candidate
    return list(samples.values())


def build(trajectories, previous, evaluation, seed):
    if len(trajectories) < 10:
        raise ValueError('At least 10 source trajectories required')
    ids, starts = set(), set()
    for row in trajectories:
        if row['id'] in ids or row['state'] in starts:
            raise ValueError('Duplicate source id or start')
        ids.add(row['id']); starts.add(row['state'])
        state = row['state']
        if state == SOLVED or not row['solution'] or row['states'][0] != state:
            raise ValueError('Invalid source start')
        if len(row['states']) != len(row['solution']) + 1:
            raise ValueError('Invalid source states length')
        for index, action in enumerate(row['solution']):
            if state == SOLVED:
                raise ValueError('Source continues after solved')
            state = move(state, action)
            if state != row['states'][index + 1]:
                raise ValueError('Source transition mismatch')
        if state != SOLVED:
            raise ValueError('Source does not solve')
    distances = distances_to_goal()
    reserved = {r['state'] for r in make_cases(evaluation,17) if r['split']=='heldout'}
    previous_states = {r['state'] for r in previous}
    if reserved & previous_states:
        raise ValueError('Previous training overlaps frozen validation starts')
    shuffled = list(trajectories)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    train_end, val_end = int(n*.8), int(n*.9)
    partitions = {'train':shuffled[:train_end], 'validation':shuffled[train_end:val_end], 'test':shuffled[val_end:]}
    # Preserve historical validation starts: reject whole training trajectories,
    # so their teacher continuations cannot leak those states into later stages.
    train_sources = [r for r in partitions['train'] if not reserved.intersection(r['states'][:-1])]
    expanded = {name:expand(rows,distances) for name,rows in partitions.items()}
    expanded['train'] = expand(train_sources,distances)
    trained = previous_states | {r['state'] for r in expanded['train']}
    validation = [r for r in expanded['validation'] if r['state'] not in trained | reserved]
    # Test starts must also be absent from ALL validation source trajectories,
    # including their shared suffixes. Later rollout states may still overlap.
    validation_states = {s for r in partitions['validation'] for s in r['states'][:-1]}
    test = [r for r in expanded['test'] if r['state'] not in trained | reserved | validation_states]
    samples = {'train':expanded['train'], 'validation':validation, 'test':test}
    report = {
        'seed':seed, 'source_trajectories':n,
        'assigned_trajectories':{k:len(v) for k,v in partitions.items()},
        'retained_train_trajectories':len(train_sources),
        'train_trajectories_excluded_for_old_validation':train_end-len(train_sources),
        'previous_training_states':len(previous_states), 'reserved_old_validation_starts':len(reserved),
        'unique_samples':{k:len(v) for k,v in samples.items()},
        'by_expert_remaining':{k:dict(sorted(Counter(r['expert_remaining'] for r in v).items())) for k,v in samples.items()},
        'removed_eval_starts':{'validation':len(expanded['validation'])-len(validation),'test':len(expanded['test'])-len(test)},
        'scope':'Unseen evaluation STARTS only; intermediate rollout states may overlap training. Exact-state exclusion, not symmetry exclusion.',
        'difficulty':'expert_remaining is shortest observed suffix within its source split, not optimal distance. optimal_distance is exact only for states in depth-3 BFS, null otherwise.',
        'stage_scope':'Stage files contain expert suffixes only. Previous SFT states are used for exclusions but are not automatically mixed into these files.',
        'status':'Data preparation only; existing shallow trainer/evaluator do not accept this curriculum schema.',
    }
    manifest = {name:[r['id'] for r in rows] for name,rows in partitions.items()}
    manifest['retained_train'] = [r['id'] for r in train_sources]
    return samples,report,manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('data/cube_random_1000/all.jsonl'))
    p.add_argument('--previous-data',type=Path,default=Path('data/cube_trajectories_1000.jsonl'))
    p.add_argument('--eval-data',type=Path,default=Path('data/cube_trajectories_expanded.jsonl'))
    p.add_argument('--output',type=Path,default=Path('data/curriculum_v1'))
    p.add_argument('--seed',type=int,default=20260924)
    args=p.parse_args()
    if args.output.exists():
        p.error('Output already exists; choose a new directory')
    samples,report,manifest=build(read_rows(args.data),read_rows(args.previous_data),read_rows(args.eval_data),args.seed)
    args.output.mkdir(parents=True)
    report['inputs']={k:str(getattr(args,k)) for k in ('data','previous_data','eval_data')}
    for name,rows in samples.items():
        (args.output/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    for depth in (3,5):
        rows=[r for r in samples['train'] if r['expert_remaining']<=depth]
        (args.output/f'train_upto_{depth}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    print(f'Saved {args.output}. No training performed.')


if __name__=='__main__':
    main()
