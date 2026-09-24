#!/usr/bin/env python3
"""Expand verified shortest solutions into deduplicated state/next-action samples.
Uses cube/eval.py + cube/smoke.py; no torch, GPU or Node.js required.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
from cube.eval import ACTIONS, SOLVED, move, distances_to_goal, make_cases


def expand(rows):
    if not rows:
        raise ValueError('Empty source dataset')
    distances = distances_to_goal()
    source_states = set()
    # Validate complete trajectories before making labels.
    for row in rows:
        state, solution = row['state'], row['solution']
        if (state in source_states or distances.get(state) not in (1, 2, 3)
                or len(solution) != distances[state] or row['optimal_distance'] != distances[state]
                or not solution or row['target_action'] != solution[0]
                or ACTIONS.index(solution[0]) != row['target_index']):
            raise ValueError('Invalid source state, distance or label')
        source_states.add(state)
        for a in solution:
            if a not in ACTIONS:
                raise ValueError('Invalid solution action')
            child = move(state, a)
            if distances.get(child) != distances[state] - 1:
                raise ValueError('Solution is not a shortest path')
            state = child
        if state != SOLVED:
            raise ValueError('Solution did not restore the cube')
    # Preserve all existing first-action labels; do not create conflicting labels
    # when distinct shortest solutions reach the same state.
    by_state = {r['state']: dict(r, trajectory_sources=[]) for r in rows}
    occurrences, alternatives = 0, 0
    for row in rows:
        state = row['state']
        for step, action in enumerate(row['solution']):
            occurrences += 1
            if state not in by_state:
                faces = {f: state[i*9:i*9+9] for i, f in enumerate('URFDLB')}
                remaining = row['solution'][step:]
                by_state[state] = {
                    'id': f'cube-intermediate-{len(by_state)-len(rows)+1:04d}',
                    'state': state, 'faces': faces,
                    'input': '\n'.join(f'{f}: {faces[f]}' for f in 'URFDLB') + '\nAction:',
                    'optimal_distance': distances[state], 'solution': remaining,
                    'solution_length': len(remaining), 'target_action': action,
                    'target_index': ACTIONS.index(action), 'trajectory_sources': [],
                }
            stored = by_state[state]
            alternatives += int(stored['target_action'] != action)
            stored['trajectory_sources'].append({'source_id': row['id'], 'step': step})
            state = move(state, action)
    result = list(by_state.values())
    for row in result:
        # The chosen policy's successors are also covered, even after deduplication.
        child = move(row['state'], row['target_action'])
        assert child == SOLVED or child in by_state
    return result, {
        'source_count': len(rows), 'raw_steps': occurrences, 'unique_states': len(result),
        'added_states': len(result)-len(rows), 'deduplicated_occurrences': occurrences-len(result),
        'alternative_optimal_action_occurrences': alternatives,
        'by_depth': dict(sorted(Counter(r['optimal_distance'] for r in result).items())),
        'label_policy': 'Preserve original labels; otherwise first encountered verified optimal action.',
        'validation': 'All source solutions shortest and replayed; all intermediate states supervised; solved states omitted.',
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path(__file__).resolve().parents[1] / 'data/cube_shallow_256.jsonl')
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'data/cube_trajectories_expanded.jsonl')
    args = p.parse_args()
    rows = [json.loads(s) for s in args.data.read_text().splitlines() if s.strip()]
    expanded, report = expand(rows)
    cases = make_cases(expanded, 17)
    old_cases = make_cases(rows, 17)
    new_states = {r['state'] for r in expanded}
    heldout = [c for c in cases if c['split'] == 'heldout']
    assert not {c['state'] for c in heldout} & new_states
    report['old_heldout_starts_now_in_training'] = sum(c['split'] == 'heldout' and c['state'] in new_states for c in old_cases)
    report['new_heldout_by_depth'] = dict(sorted(Counter(c['depth'] for c in heldout).items()))
    report['evaluation_seed'] = 17
    report['evaluation_scope'] = 'Unseen initial states; later states may overlap training. Compare both checkpoints using --data pointing to this expanded dataset and --seed 17.'
    outputs = {
        args.output: ''.join(json.dumps(r) + '\n' for r in expanded),
        args.output.with_suffix('.meta.json'): json.dumps(report, indent=2) + '\n',
        args.output.with_suffix('.eval_cases.jsonl'): ''.join(json.dumps(c) + '\n' for c in cases),
    }
    if any(path.exists() for path in outputs):
        p.error('Output already exists; choose another --output')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path, content in outputs.items():
        path.write_text(content)
    print(json.dumps(report, indent=2))
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
