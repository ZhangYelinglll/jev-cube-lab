// Run: node web/tools/generate_shallow.mjs
// Exact shallow-state BFS solver using cubejs transitions; labels depend only on state.
import Cube from 'cubejs';
import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';

const faces = 'URFDLB';
const actions = [...faces].flatMap(f => [f, `${f}'`, `${f}2`]);
const quotas = [0, 18, 100, 138];
const seed = 20260924;
let rng = seed;
function random() {
  rng = (Math.imul(1664525, rng) + 1013904223) >>> 0;
  return rng / 2 ** 32;
}
const solved = new Cube().asString();
const distances = new Map([[solved, 0]]);
let frontier = [{ state: solved, scramble: [] }];
const selected = [];
const population = {};
for (let depth = 1; depth <= 3; depth++) {
  const next = [];
  for (const parent of frontier) {
    for (const move of actions) {
      const state = Cube.fromString(parent.state).move(move).asString();
      if (distances.has(state)) continue;
      distances.set(state, depth);
      next.push({ state, scramble: [...parent.scramble, move], distance: depth });
    }
  }
  population[depth] = next.length;
  // BFS visits every shallower state first, so distance is exact in HTM.
  frontier = next;
  const shuffled = [...next];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  assert(shuffled.length >= quotas[depth]);
  selected.push(...shuffled.slice(0, quotas[depth]));
}
// Known complete shallow shells are an independent guard on BFS/metric.
assert.deepEqual(population, { 1: 18, 2: 243, 3: 3240 });
// Solve from the state alone by descending the complete exact-distance table.
// No scramble or inverse-scramble sequence is given to this solver.
function solveShallow(state) {
  assert(distances.has(state), 'State outside the depth-3 solver table');
  const solution = [];
  while (state !== solved) {
    const move = actions.find(action => {
      const child = Cube.fromString(state).move(action).asString();
      return distances.get(child) === distances.get(state) - 1;
    });
    assert(move, 'Missing shortest-path edge');
    solution.push(move);
    state = Cube.fromString(state).move(move).asString();
  }
  return solution;
}
const records = selected.map((sample, index) => {
  const solution = solveShallow(sample.state);
  assert(solution.every(m => actions.includes(m)));
  assert.equal(solution.length, sample.distance);
  assert.equal(new Cube().move(sample.scramble.join(' ')).asString(), sample.state);
  assert(Cube.fromString(sample.state).move(solution.join(' ')).isSolved());
  const stateFaces = Object.fromEntries([...faces].map((f, i) => [f, sample.state.slice(i * 9, i * 9 + 9)]));
  return {
    id: `cube-shallow-${String(index + 1).padStart(4, '0')}`,
    state: sample.state,
    faces: stateFaces,
    input: [...faces].map(f => `${f}: ${stateFaces[f]}`).join('\n') + '\nAction:',
    scramble: sample.scramble,
    scramble_length: sample.scramble.length,
    optimal_distance: sample.distance,
    solution,
    solution_length: solution.length,
    target_action: solution[0],
    target_index: actions.indexOf(solution[0]),
  };
});
assert.equal(records.length, 256);
assert.equal(new Set(records.map(r => r.state)).size, 256);
const output = new URL('../../data/', import.meta.url);
await mkdir(output, { recursive: true });
await writeFile(new URL('cube_shallow_256.jsonl', output), records.map(r => JSON.stringify(r)).join('\n') + '\n');
await writeFile(new URL('cube_shallow_256.meta.json', output), JSON.stringify({
  count: records.length, seed, solver: 'Exact breadth-first distance-table search, depth <= 3; cubejs 1.3.2 transitions',
  metric: 'HTM: quarter turn, inverse quarter turn, and half turn each cost one move',
  face_order: faces, facelets: 'Each face row-major, viewed from outside; cubejs convention',
  actions, samples_by_optimal_distance: { 1: 18, 2: 100, 3: 138 },
  complete_shell_sizes: population,
  validation: '256 distinct non-solved states; all solver solutions replay to solved and match BFS shortest distance',
  usage: 'Training smoke/overfit set only. Feed input (or faces); labels are target_index. Never feed scramble, solution, or distance to the model. Multiple optimal solutions may exist; one verified solution is stored. Not a held-out evaluation set.',
}, null, 2) + '\n');
console.log(`PASS: ${records.length} unique states, exact distances 1-3, all solver solutions verified.`);
console.log(new URL('cube_shallow_256.jsonl', output).pathname);
