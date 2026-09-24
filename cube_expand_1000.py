#!/usr/bin/env python3
"""Extend the 323-state set to 1000, preserving its 332 held-out starts.
Run with Python only; requires cube_eval.py and cube_smoke.py beside this file.
"""
import json
import random
from collections import Counter
from pathlib import Path
from cube_eval import ACTIONS, SOLVED, distances_to_goal, make_cases, move


def build(source):
    distances = distances_to_goal()
    by_state = {r['state']: dict(r) for r in source}
    if len(source) != 323 or len(by_state) != 323:
        raise ValueError('Expected the existing 323 unique training states')
    frozen_cases = make_cases(source, 17)
    heldout = {c['state'] for c in frozen_cases if c['split'] == 'heldout'}
    assert len(heldout) == 332 and not heldout & by_state.keys()
    # All remaining depth-2 states are already evaluation starts. New depth-3
    # samples must therefore have an optimal continuation through trained states.
    candidates = []
    for state, depth in distances.items():
        if depth != 3 or state in by_state or state in heldout:
            continue
        for action in ACTIONS:
            child = move(state, action)
            if distances.get(child) == 2 and child in by_state:
                candidates.append((state, [action] + by_state[child]['solution']))
                break
    random.Random(20260924).shuffle(candidates)
    if len(candidates) < 677:
        raise ValueError('Not enough eligible states without held-out overlap')
    for i, (state, solution) in enumerate(candidates[:677], 1):
        faces = {f: state[j*9:j*9+9] for j, f in enumerate('URFDLB')}
        by_state[state] = {
            'id': f'cube-extra-1000-{i:04d}', 'state': state, 'faces': faces,
            'input': '\n'.join(f'{f}: {faces[f]}' for f in 'URFDLB') + '\nAction:',
            'solution': solution, 'solution_length': len(solution),
            'optimal_distance': 3, 'target_action': solution[0],
            'target_index': ACTIONS.index(solution[0]),
        }
    result = list(by_state.values())
    assert len(result) == 1000 and len({r['id'] for r in result}) == 1000
    assert not heldout & by_state.keys()
    # Validate EVERY full trajectory, not merely its initial state.
    for row in result:
        state = row['state']
        assert row['target_index'] == ACTIONS.index(row['target_action'])
        assert row['target_action'] == row['solution'][0]
        assert len(row['solution']) == distances[state] == row['optimal_distance']
        for action in row['solution']:
            assert state in by_state and state not in heldout
            child = move(state, action)
            assert distances[child] == distances[state] - 1
            state = child
        assert state == SOLVED
    for row in source:
        assert by_state[row['state']] == row  # Original rows and labels preserved.
    report = {
        'count': 1000, 'retained_source_states': 323, 'added_states': 677,
        'depth_counts': dict(sorted(Counter(r['optimal_distance'] for r in result).items())),
        'seed': 20260924, 'eligible_new_depth3_states': len(candidates),
        'heldout_starts': len(heldout), 'training_heldout_overlap': 0,
        'trajectory_heldout_overlap': 0,
        'solver': 'Exact depth-3 BFS distances with shortest continuation through original training states',
        'sampling_scope': 'New depth-3 starts are selected to have an optimal path through the original 323-state set; not uniform over every depth-3 state.',
        'evaluation': 'Use cube_eval.py --data data/cube_trajectories_expanded.jsonl --seed 17 for BOTH old and new checkpoints. This freezes the same 332 held-out starts and original 323-state diagnostic pool.',
        'validation': 'All 1000 solutions replay to solved in exact shortest distance; all intermediate states supervised; original rows unchanged.',
    }
    return result, report


def main():
    folder = Path(__file__).parent / 'data'
    source = [json.loads(s) for s in (folder / 'cube_trajectories_expanded.jsonl').read_text().splitlines()]
    rows, report = build(source)
    output = folder / 'cube_trajectories_1000.jsonl'
    meta = output.with_suffix('.meta.json')
    if output.exists() or meta.exists():
        raise FileExistsError('Refusing to overwrite existing 1000-state data')
    output.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    meta.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    print(f'Saved {output}')


if __name__ == '__main__':
    main()
