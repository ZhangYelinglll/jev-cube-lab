"""Run with python -m tests.check_eval; no torch or GPU required."""
import json
from pathlib import Path
from cube.smoke import STATES
from cube.eval import ACTIONS, SOLVED, move, distances_to_goal, rollout, summarize, make_cases


def main():
    for a, expected in STATES.items():
        assert move(SOLVED, a) == expected
    rows = [json.loads(line) for line in (Path(__file__).resolve().parents[1] / 'data/cube_shallow_256.jsonl').read_text().splitlines()]
    selected = make_cases(rows, 17)
    heldout = [c for c in selected if c['split'] == 'heldout']
    assert len(selected) == 655 and len(heldout) == 399
    assert not {c['state'] for c in heldout} & {r['state'] for r in rows}
    assert selected == make_cases(rows, 17)
    distances = distances_to_goal()
    assert [sum(d == i for d in distances.values()) for i in range(4)] == [1, 18, 243, 3240]
    cases = [{'id':str(i),'state':s,'depth':d,'split':'test'}
             for i,(s,d) in enumerate(distances.items()) if d > 0]
    def oracle(states):
        return [next(i for i,a in enumerate(ACTIONS) if distances.get(move(s,a)) == distances[s]-1) for s in states]
    solved = rollout(cases, oracle, 10)
    assert all(r['solved'] and len(r['moves']) == r['depth'] for r in solved)
    # Constant U must cycle and still run to budget, with no anti-loop filters.
    stuck = rollout([{'id':'loop','state':move(SOLVED,'R'),'depth':1,'split':'test'}], lambda states:[0]*len(states), 10)
    assert not stuck[0]['solved'] and stuck[0]['repeated_states'] > 0 and len(stuck[0]['moves']) == 10
    assert summarize(stuck)['success_rate'] == 0 and summarize(stuck)['mean_steps_success'] is None
    last = rollout([{'id':'last','state':move(SOLVED,'U'),'depth':1,'split':'test'}], lambda states:[1]*len(states), 1)
    assert last[0]['solved']
    print('PASS: all 3501 shallow states solved by oracle; cycles continue to budget; last-step success counted')


if __name__ == '__main__':
    main()
