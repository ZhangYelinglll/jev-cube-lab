import assert from 'node:assert/strict';
import { Cube, MOVES, faceState, scramble, outcome, decisionInput } from './state.js';
for (const move of MOVES) {
  const cube = new Cube();
  cube.move(move);
  assert(!cube.isSolved());
  cube.move(Cube.inverse(move));
  assert(cube.isSolved());
}
assert.deepEqual(faceState(new Cube()).U, Array(9).fill('U'));
for (const depth of [1, 2, 3, 10, 20]) {
  const moves = scramble(depth);
  assert.equal(moves.length, depth);
  assert(!new Cube().move(moves.join(' ')).isSolved());
}
const cube = new Cube().move('R');
assert.equal(outcome(cube, 2, 50), null);
assert.equal(outcome(cube, 50, 50), 'limit');
assert.equal(outcome(new Cube(), 50, 50), 'solved');
// Verify move/inverse invariants in the renderer's cube model too.
const { cube3x3x3 } = await import('cubing/puzzles');
const kpuzzle = await cube3x3x3.kpuzzle();
for (const move of MOVES) {
  const initial = kpuzzle.defaultPattern();
  assert(initial.applyAlg(`${move} ${Cube.inverse(move)}`).isIdentical(initial));
}
console.log('Cube state checks passed');

const input = decisionInput(cube, [{choice:'R', input:{faces:faceState(new Cube())}, after:faceState(cube)}]);
assert.equal(input.trajectory.steps[0].faces_after, cube.asString());
assert.equal(input.trajectory.initial_faces, new Cube().asString());
assert.deepEqual(input.history,['R']);

// Repeated quarter turns are legal until they revisit an actual prior state.
const turning = new Cube().move('R');
const steps = [];
for (let i = 0; i < 3; i++) {
  const before = decisionInput(turning, steps);
  assert(!before.repeated_moves.includes('U'));
  turning.move('U');
  steps.push({choice:'U', input:before, after:faceState(turning)});
}
const fourth = decisionInput(turning, steps);
assert(fourth.repeated_moves.includes('U'));
assert(fourth.repeated_moves.includes("U'"));
assert.equal(steps[1].choice, 'U');
console.log('Consecutive U and repeated-state prediction checks passed');
