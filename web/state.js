// Deliberately import only the state engine: no solver is loaded.
import Cube from 'cubejs/lib/cube.js';
export { Cube };
export const MOVES = [...'URFDLB'].flatMap(f => ['', "'", '2'].map(s => f + s));
export function faceState(cube) {
  const stickers = cube.asString();
  return Object.fromEntries([...'URFDLB'].map((f, i) => [f, [...stickers.slice(i * 9, i * 9 + 9)]]));
}
export function scramble(depth) {
  let result;
  do {
    result = [];
    for (let i = 0; i < depth; i++) {
      const options = MOVES.filter(m => m[0] !== result.at(-1)?.[0]);
      result.push(options[Math.floor(Math.random() * options.length)]);
    }
  } while (new Cube().move(result.join(' ')).isSolved());
  return result;
}
export function outcome(cube, count, limit) {
  if (cube.isSolved()) return 'solved';
  if (count >= limit) return 'limit';
  return null;
}

export function decisionInput(cube, steps) {
  const faces = faceState(cube);
  const encode = state => [...'URFDLB'].map(f => state[f].join('')).join('');
  const trajectory = {
    initial_faces:encode(steps[0]?.input.faces || faces),
    steps:steps.map(s => ({move:s.choice, faces_after:encode(s.after)})),
  };
  const seen = new Set([trajectory.initial_faces, ...trajectory.steps.map(s => s.faces_after)]);
  const repeated_moves = MOVES.filter(move => seen.has(cube.clone().move(move).asString()));
  return {faces, history:steps.map(s => s.choice), trajectory, repeated_moves};
}
