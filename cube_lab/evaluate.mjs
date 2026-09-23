// Live smoke evaluation; calls the running server and consumes TypeSafe usage.
import {mkdir, writeFile} from 'node:fs/promises';
import {Cube, faceState, outcome, decisionInput} from './state.js';
const results=[];
for (const setup of ['U','R',"F'",'R U','R U F']) {
  const cube = new Cube().move(setup);
  const run = {setup, initial:faceState(cube), steps:[], status:null};
  while (!run.status) {
    const input = decisionInput(cube,run.steps);
    const response = await fetch('http://127.0.0.1:8765/api/step',{
      method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(input)});
    const result = await response.json();
    if (!response.ok) {run.status='error';run.error=result.error;break;}
    cube.move(result.choice);
    run.steps.push({...result,input,after:faceState(cube)});
    run.status = outcome(cube,run.steps.length,10);
    console.log(setup,run.steps.length,result.choice,run.status || 'continue');
  }
  results.push(run);
}
await mkdir('../data', {recursive:true});
await writeFile('../data/jev_cube_smoke.json',JSON.stringify({date:new Date().toISOString(),limit:10,results},null,2));
console.log('SUMMARY',JSON.stringify(results.map(r=>({setup:r.setup,status:r.status,moves:r.steps.map(s=>s.choice)}))));
