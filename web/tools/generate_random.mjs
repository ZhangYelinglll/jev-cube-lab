// Random-walk dataset; solution lengths are solver output lengths, NOT optimal distances.
import Cube from 'cubejs';
import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve, join } from 'node:path';

const output = resolve(process.argv[2] || 'data/cube_random_1000');
const count = 1000, scrambleLength = 30, seed = 20260925;
const faces = 'URFDLB';
const actions = [...faces].flatMap(f => [f, `${f}'`, `${f}2`]);
let rng = seed;
function randomInt(n) {
  rng = (Math.imul(1664525, rng) + 1013904223) >>> 0;
  return Math.floor(rng / 2 ** 32 * n);
}
// Refuse to overwrite a previous dataset.
await mkdir(output, { recursive: false });
Cube.initSolver();
const solved = new Cube().asString();
const seen = new Set(), records = [], buckets = new Map();
for (let attempt = 0; records.length < count; attempt++) {
  assert(attempt < count * 10, 'Too many duplicate states');
  const scramble = [];
  for (let j = 0; j < scrambleLength; j++) {
    const choices = actions.filter(a => a[0] !== scramble.at(-1)?.[0]);
    scramble.push(choices[randomInt(choices.length)]);
  }
  const cube = new Cube().move(scramble.join(' '));
  const state = cube.asString();
  if (state === solved || seen.has(state)) continue;
  const solution = cube.solve(22).trim().split(/\s+/).filter(Boolean);
  assert(solution.length > 0 && solution.length <= 22);
  assert(solution.every(a => actions.includes(a)));
  const replay = Cube.fromString(state), states = [state];
  for (const action of solution) {
    replay.move(action);
    states.push(replay.asString());
  }
  assert(replay.isSolved(), 'Solver solution failed replay');
  assert.equal(new Cube().move(scramble.join(' ')).asString(), state);
  const record = {
    id: `cube-random-${String(records.length + 1).padStart(4, '0')}`,
    state, faces: Object.fromEntries([...faces].map((f,i) => [f,state.slice(i*9,i*9+9)])),
    scramble, scramble_length: scramble.length, solution, solution_length: solution.length,
    optimal_distance: null, states,
  };
  records.push(record);
  seen.add(state);
  if (!buckets.has(solution.length)) buckets.set(solution.length, []);
  buckets.get(solution.length).push(record);
  if (records.length % 50 === 0) console.log(`Verified ${records.length}/${count}`);
}
const jsonl = rows => rows.map(r => JSON.stringify(r)).join('\n') + '\n';
await writeFile(join(output, 'all.jsonl'), jsonl(records));
await mkdir(join(output, 'by_length'));
const counts = {};
for (const [length, rows] of [...buckets].sort((a,b) => a[0]-b[0])) {
  counts[length] = rows.length;
  await writeFile(join(output, 'by_length', `${length}.jsonl`), jsonl(rows));
}
const metadata = {
  count, seed, scramble_length: scrambleLength,
  generation: 'Seeded 30-move random walks, no consecutive same-face turns; not uniform random states',
  solver: 'cubejs 1.3.2 two-phase solver, maxDepth=22',
  metric: 'HTM: quarter turn, inverse turn and half turn each count as one action',
  grouping: 'Actual returned solution length, not shortest distance',
  counts_by_solution_length: counts,
  mean_solution_length: records.reduce((s,r) => s+r.solution_length,0)/count,
  validation: '1000 unique non-solved starts; all scrambles and full solutions replayed; states include start and terminal',
  usage: 'Raw expert trajectories; no train/test split or exclusion applied; do not feed solution or states[1:] as model input',
};
await writeFile(join(output, 'meta.json'), JSON.stringify(metadata,null,2)+'\n');
console.log(JSON.stringify(metadata,null,2));
