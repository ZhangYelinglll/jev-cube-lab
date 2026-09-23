"""Offline boundary checks: uv run python check_cube_lab.py."""
from cube_server import MOVES, make_payload, validate_answer

faces = {f: [f] * 9 for f in 'URFDLB'}
payload = make_payload({'faces': faces, 'history': [], 'scramble': 'SECRET'})
assert 'SECRET' not in str(payload)
assert len(payload['questions']['move']['criteria']) == 18
assert payload['state']['faces'] == faces
answer = {'type': 'choice', 'choice': 'R', 'confidence': 1.0,
          'probabilities': {m: float(m == 'R') for m in MOVES}}
assert validate_answer({'model': 'test', 'answers': {'move': answer}})['choice'] == 'R'
for bad in ({'faces': {}, 'history': []}, {'faces': faces, 'history': ['X']},
            {'faces': {**faces, 'U': ['R'] * 9}, 'history': []}):
    try:
        make_payload(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid cube request accepted')
for patch in ({'choice': 'X'}, {'confidence': float('nan')}, {'probabilities': {'R': 1}},
              {'probabilities': {m: 0 for m in MOVES}}):
    try:
        validate_answer({'model': 'test', 'answers': {'move': {**answer, **patch}}})
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid model response accepted')

trace = {'initial_faces': ''.join(f * 9 for f in 'URFDLB'), 'steps': []}
assert make_payload({'faces':faces,'history':[],'trajectory':trace})['state']['trajectory'] == trace

for trace in ({'initial_faces':'bad','steps':[]}, {'initial_faces':trace['initial_faces'],'steps':[{'move':'R','faces_after':'bad'}]}):
    try:
        make_payload({'faces':faces,'history':[],'trajectory':trace})
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid trajectory accepted')
# All 49 previous steps fit within the request limit, with no history truncation.
state = ''.join(f * 9 for f in 'URFDLB')
data = {'faces':faces,'history':['R'] * 49,'trajectory':{'initial_faces':state,'steps':[{'move':'R','faces_after':state}] * 49}}
assert len(make_payload(data)['state']['trajectory']['steps']) == 49
import json
assert len(json.dumps(data).encode()) < 16384
print('Cube API and trajectory checks passed')

# Exclude only the immediate inverse; repeated-state exclusions come from simulation.
for last in MOVES:
    inverse = last if last.endswith('2') else last[0] if last.endswith("'") else last + "'"
    request = {'faces':faces, 'history':[last], 'trajectory':{
        'initial_faces':state, 'steps':[{'move':last,'faces_after':state}]}}
    options = make_payload(request)['questions']['move']['criteria']
    assert set(options) == set(MOVES) - {inverse}
    chosen = next(iter(options))
    valid = {**answer, 'choice':chosen, 'probabilities':{m:float(m == chosen) for m in options}}
    assert validate_answer({'model':'test','answers':{'move':valid}}, options)['choice'] == chosen
    try:
        validate_answer({'model':'test','answers':{'move':{**valid,'choice':inverse}}}, options)
    except ValueError:
        pass
    else:
        raise AssertionError('Forbidden inverse accepted')
print('All 18 inverse exclusions passed')

request = {'faces':faces, 'history':['U'], 'trajectory':{
    'initial_faces':state, 'steps':[{'move':'U','faces_after':state}]}, 'repeated_moves':['U', "U'"]}
assert 'U' not in make_payload(request)['questions']['move']['criteria']
request['repeated_moves'] = MOVES
# Never end the experiment just because all moves revisit prior states.
assert len(make_payload(request)['questions']['move']['criteria']) == 17
assert make_payload(request)['state']['repeat_filter_relaxed'] is True
print('Repeated-state filtering and fallback passed')
