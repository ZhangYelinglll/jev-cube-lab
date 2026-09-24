"""Replay published random trajectories and verify their length buckets; no dependencies."""
import json
from collections import Counter
from pathlib import Path
from cube.eval import SOLVED, move

folder = Path(__file__).resolve().parents[1] / 'data/cube_random_1000'
rows = [json.loads(line) for line in (folder / 'all.jsonl').read_text().splitlines()]
meta = json.loads((folder / 'meta.json').read_text())
assert len(rows) == len({r['state'] for r in rows}) == len({r['id'] for r in rows}) == 1000
for row in rows:
    state = SOLVED
    assert len(row['scramble']) == row['scramble_length'] == 30
    for action in row['scramble']:
        state = move(state, action)
    assert state == row['state'] == row['states'][0] and state != SOLVED
    assert row['optimal_distance'] is None
    assert len(row['solution']) == row['solution_length'] == len(row['states']) - 1
    for i, action in enumerate(row['solution']):
        state = move(state, action)
        assert state == row['states'][i + 1]
    assert state == SOLVED
counts = dict(Counter(str(r['solution_length']) for r in rows))
assert counts == meta['counts_by_solution_length']
assert meta['count'] == len(rows)
assert abs(meta['mean_solution_length'] - sum(r['solution_length'] for r in rows)/len(rows)) < 1e-9
bucket_rows = []
for path in (folder / 'by_length').glob('*.jsonl'):
    part = [json.loads(line) for line in path.read_text().splitlines()]
    assert all(r['solution_length'] == int(path.stem) for r in part)
    bucket_rows.extend(part)
assert sorted(bucket_rows, key=lambda r:r['id']) == sorted(rows, key=lambda r:r['id'])
print('PASS: 1000 distinct starts, scrambles, all solution steps, terminal states, metadata and exact buckets')
