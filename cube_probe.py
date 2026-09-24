#!/usr/bin/env python3
"""Compare exploration settings on identical non-test starts without training."""
import argparse
import json
import math
import random
import time
from pathlib import Path
import torch
from cube_grpo import collect_groups, load_agent
from cube_eval import distances_to_goal, make_cases
from cube_train import read_data


def summarize(groups):
    valid = [g for g in groups if not any(e['invalid'] for e in g)]
    mixed = [g for g in valid if 0 < sum(e['outcome'] for e in g) < len(g)]
    def average(values):
        return sum(values)/len(values) if values else None
    episodes = [e for g in valid for e in g]
    first = [e['steps'][0] for g in groups for e in g if e['steps']]
    return {
        'attempted_groups':len(groups), 'accepted_groups':len(valid),
        'discarded_groups':len(groups)-len(valid), 'informative_groups':len(mixed),
        'informative_fraction_attempted':len(mixed)/len(groups) if groups else None,
        'informative_fraction_accepted':len(mixed)/len(valid) if valid else None,
        'all_success_groups':sum(all(e['outcome'] for e in g) for g in valid),
        'all_failure_groups':sum(not any(e['outcome'] for e in g) for g in valid),
        'accepted_success_rate':average([e['outcome'] for e in episodes]),
        'start_entropy_nats':average([r['entropy_nats'] for r in first]),
        'start_top1_probability':average([r['top1_probability'] for r in first]),
        'mean_unique_first_actions':average([len({e['steps'][0]['action'] for e in g if e['steps']}) for g in groups]),
        'mean_unique_trajectories_accepted':average([len({tuple(r['action'] for r in e['steps']) for e in g}) for g in valid]),
        'observed_actions':sum(len(e['steps']) for g in groups for e in g),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--eval-data',type=Path,default=Path('data/cube_trajectories_expanded.jsonl'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--starts',type=int,default=64)
    p.add_argument('--temperatures',type=float,nargs='+',default=[1.0,1.3,1.6])
    p.add_argument('--group-sizes',type=int,nargs='+',default=[8,16])
    p.add_argument('--max-steps',type=int,default=10)
    p.add_argument('--seed',type=int,default=17)
    args=p.parse_args()
    if args.starts < 1 or args.max_steps < 1 or min(args.group_sizes)<2:
        p.error('Positive counts and group sizes >= 2 required')
    if any(not math.isfinite(t) or t<=0 for t in args.temperatures):
        p.error('Temperatures must be finite and positive')
    if args.output.exists():
        p.error('Choose a new output directory')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        p.error('CUDA BF16 GPU required')
    reserved={c['state'] for c in make_cases(read_data(args.eval_data),17) if c['split']=='heldout'}
    training=read_data(args.checkpoint.parent/'training_data.jsonl')
    if reserved & {r['state'] for r in training}:
        raise ValueError('Training data overlaps reserved test starts')
    distances=distances_to_goal()
    pool=sorted(s for s,d in distances.items() if d in (2,3) and s not in reserved)
    if args.starts>len(pool):
        p.error('Too many starts')
    starts=random.Random(args.seed).sample(pool,args.starts)
    args.output.mkdir(parents=True)
    (args.output/'run.json').write_text(json.dumps({**vars(args),'start_states':starts,
        'reserved_count':len(reserved),'updates':0,
        'scope':'Frozen actor probe; confidence is not evaluated. Invalid trajectories stop at reserved states.',
        'comparison':'Smaller groups are prefixes of the same maximum-size rollouts; compare informative groups per attempted group and per action.'},default=str,indent=2))
    torch.manual_seed(args.seed)
    policy=load_agent(args.checkpoint,args.max_steps,'cuda').eval()
    results=[]
    with (args.output/'groups.jsonl').open('w') as trace:
        for temperature in args.temperatures:
            groups=[]
            torch.manual_seed(args.seed)
            started=time.monotonic()
            for index,state in enumerate(starts):
                # One start at a time keeps inference memory independent of --starts.
                group=collect_groups(policy,[state],max(args.group_sizes),args.max_steps,
                                     reserved,temperature=temperature,return_all=True)[0]
                groups.append(group)
                trace.write(json.dumps({'temperature':temperature,'start_index':index,'depth':distances[state],
                    'episodes':group})+'\n')
                trace.flush()
                if (index+1)%8==0:
                    print(f'T={temperature}: {index+1}/{len(starts)} starts',flush=True)
            for size in args.group_sizes:
                row={'temperature':temperature,'group_size':size,
                     'collection_seconds_max_group':round(time.monotonic()-started,2),
                     **summarize([g[:size] for g in groups])}
                row['informative_groups_per_1000_actions']=1000*row['informative_groups']/max(1,row['observed_actions'])
                results.append(row)
                print(json.dumps(row),flush=True)
    (args.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    print(f'Saved {args.output}/summary.json; no weights updated.',flush=True)


if __name__=='__main__':
    main()
