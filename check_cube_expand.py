"""No GPU needed: verify full-trajectory expansion and unchanged original labels."""
import json
from pathlib import Path
from cube_eval import SOLVED, move, distances_to_goal
from cube_expand import expand

rows = [json.loads(s) for s in (Path(__file__).parent / 'data/cube_shallow_256.jsonl').read_text().splitlines()]
expanded, report = expand(rows)
lookup = {r['state']: r for r in expanded}
assert len(lookup) == len(expanded) > len(rows)
for row in rows:
    assert lookup[row['state']]['target_action'] == row['target_action']
for row in expanded:
    state = row['state']
    for a in row['solution']:
        assert state in lookup, 'Trajectory contains an unsupervised intermediate state'
        state = move(state, a)
    assert state == SOLVED
assert SOLVED not in lookup
# Taking any stored next-action must remain in the training-state closure.
for row in expanded:
    after = move(row['state'], row['target_action'])
    assert after == SOLVED or after in lookup
assert report['raw_steps'] == sum(len(r['solution']) for r in rows)
bad = dict(rows[0], solution=['R'] * 3)
try:
    expand([bad])
except ValueError:
    pass
else:
    raise AssertionError('Invalid solution accepted')
print('PASS: originals preserved, all solution steps supervised, no terminal label, invalid solution rejected')
print(json.dumps(report, indent=2))
