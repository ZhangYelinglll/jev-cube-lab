// Offline UI checks. BROWSER_BIN may select an existing Chromium installation.
import {chromium} from 'playwright';
import assert from 'node:assert/strict';
const browser = await chromium.launch({headless:true, executablePath:process.env.BROWSER_BIN, args:['--no-sandbox']});
const page = await browser.newPage({viewport:{width:1360,height:1050}});
const errors=[];
page.on('pageerror',e=>errors.push(e.message));
await page.addInitScript(()=>{Math.random=()=>0;Object.defineProperty(crypto,'randomUUID',{value:undefined});}); // 1-step scramble is U.
await page.route('**/api/config',route=>route.fulfill({json:{ready:true,model:'offline-test'}}));
let requests=0, move="U'", fail=false, delay=0, networkFailures=0;
await page.route('**/api/step',async route=>{
  requests++;
  const input=route.request().postDataJSON();
  assert.deepEqual(Object.keys(input).sort(),['faces','history','repeated_moves','trajectory']);
  assert.equal(Object.values(input.faces).flat().length,54);
  assert.deepEqual(input.trajectory.steps.map(s=>s.move),input.history);
  assert.equal(input.trajectory.steps.at(-1)?.faces_after || input.trajectory.initial_faces, [...'URFDLB'].map(f=>input.faces[f].join('')).join(''));
  if (delay) await new Promise(r=>setTimeout(r,delay));
  if (networkFailures > 0) {networkFailures--; return route.fulfill({status:502,json:{error:'测试连接超时',retryable:true}});}
  if (fail) return route.fulfill({status:502,json:{error:'测试网络错误'}});
  const last=input.history.at(-1);
  const inverse=last ? (last.endsWith('2') ? last : last.endsWith("'") ? last[0] : last+"'") : null;
  const available=[...'URFDLB'].flatMap(f=>['',"'",'2'].map(s=>f+s)).filter(m=>m!==inverse);
  const fresh=available.filter(m=>!input.repeated_moves.includes(m));
  const moves=fresh.length ? fresh : available;
  const chosen=moves.includes(move) ? move : moves.find(m=>m[0]!=='U');
  assert(moves.includes(chosen));
  await route.fulfill({json:{type:'choice',choice:chosen,confidence:1,probabilities:Object.fromEntries(moves.map(m=>[m,Number(m===chosen)])),model:'offline-test',elapsed_ms:123}});
});
await page.goto(process.env.BASE_URL || 'http://127.0.0.1:8765');
await page.waitForFunction(()=>document.getElementById('message').textContent !== '正在加载魔方组件…');
assert.doesNotMatch(await page.locator('#message').innerText(), /加载失败/);
await page.waitForFunction(()=>!document.getElementById('new').disabled);
await page.locator('#depth').selectOption('1');
await page.waitForTimeout(800);
assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
await page.locator('#step').click();
await page.waitForFunction(()=>document.getElementById('phase').textContent==='还原成功');
assert.equal(await page.locator('#history-count').innerText(),'1');
// The rendered state must also be solved, not just the logical state.
assert(await page.evaluate(async()=>{
  const player=document.querySelector('twisty-player');
  const pattern=await player.experimentalModel.currentPattern.get();
  return pattern.experimentalIsSolved({ignorePuzzleOrientation:true,ignoreCenterOrientation:true});
}));
await page.locator('#replay').click();
await page.waitForFunction(()=>!document.getElementById('new').disabled);
assert.equal(requests,1);
assert.match(await page.locator('#stats').innerText(),/1\/1/);
// Pause finishes the one in-flight request and starts no further request.
await page.locator('#limit').selectOption('10'); move='R';delay=300;
await page.locator('#auto').click();await page.locator('#pause').click();
await page.waitForFunction(()=>!document.getElementById('new').disabled);
assert.equal(await page.locator('#history-count').innerText(),'1');
assert.equal(requests,2);
// Consecutive R moves remain legal; repeat filtering must still reach the step limit.
delay=0;
await page.locator('#auto').click();
await page.waitForFunction(()=>document.getElementById('phase').textContent==='达到步数上限');
assert.equal(await page.locator('#history-count').innerText(),'10');
assert.equal(requests,11);
assert((await page.locator('#probabilities').getByText('禁选',{exact:true}).count()) >= 1);
assert.deepEqual(await page.locator('.move-chip b').allTextContents().then(m=>m.slice(0,2)), ['R','R']);
assert.match(await page.locator('#stats').innerText(),/1\/2/);
await page.locator('.move-chip').first().click();
assert.match(await page.locator('#message').innerText(),/回看第 1 步/);
await page.screenshot({path:'/tmp/jev-cube-desktop-checked.png',fullPage:true});
// An upstream failure cannot append or animate a fake move.
await page.locator('#new').click();fail=true;
await page.locator('#step').click();
await page.waitForFunction(()=>document.getElementById('phase').textContent==='请求失败');
assert.equal(await page.locator('#history-count').innerText(),'0');
assert.equal(await page.locator('#step').isEnabled(),true);
assert.match(await page.locator('#message').innerText(),/测试网络错误/);
fail=false;move="U'";
await page.locator('#step').click();
await page.waitForFunction(()=>document.getElementById('phase').textContent==='还原成功');
// A transient network failure retries the same move; it must execute only once.
await page.locator('#new').click();networkFailures=1;
const beforeRetry=requests;
await page.locator('#auto').click();
await page.waitForFunction(()=>['还原成功','请求失败'].includes(document.getElementById('phase').textContent));
assert.equal(await page.locator('#phase').innerText(),'还原成功');
assert.equal(requests-beforeRetry,2);
assert.equal(await page.locator('#history-count').innerText(),'1');
// Persistent failures stop after three requests and never change the cube.
await page.locator('#new').click();networkFailures=10;
const beforeFailure=requests;
await page.locator('#auto').click();
await page.waitForFunction(()=>document.getElementById('phase').textContent==='请求失败');
assert.equal(requests-beforeFailure,3);
assert.equal(await page.locator('#history-count').innerText(),'0');
networkFailures=0;
await page.setViewportSize({width:390,height:844});
await page.screenshot({path:'/tmp/jev-cube-mobile-checked.png',fullPage:true});
assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
const config=await page.request.get('http://127.0.0.1:8765/.env');assert.equal(config.status(),404);
console.log('Browser errors:',errors);assert.deepEqual(errors,[]);
await browser.close();
console.log('Browser checks passed: solved, render state, replay, pause, repeated states through limit, trajectory, error/retry, mobile.');
