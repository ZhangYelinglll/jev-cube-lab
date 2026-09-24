import { TwistyPlayer } from 'cubing/twisty';
import { Cube, MOVES, faceState, scramble, outcome, decisionInput } from './state.js';

const $ = id => document.getElementById(id);
const labels = {ready:'准备就绪', running:'实验进行中', paused:'已暂停', solved:'还原成功', cycle:'重复状态', limit:'达到步数上限', abandoned:'提前结束', error:'请求失败'};
const terminal = new Set(['solved', 'limit', 'abandoned']);
const storageKey = 'jev-cube-lab-v1';
let archive = [];
try { archive = JSON.parse(localStorage.getItem(storageKey) || '[]'); if (!Array.isArray(archive)) archive = []; } catch { /* A disabled store must not stop the experiment. */ }
let cube, run, player, busy = false, automatic = false, ready = false, pauseRequested = false;

function message(text, error = false) { $('message').textContent = text; $('message').classList.toggle('error', error); }
function controls() {
  const done = !run || terminal.has(run.status);
  for (const id of ['new', 'depth', 'limit']) $(id).disabled = busy;
  $('step').disabled = !ready || busy || done;
  $('auto').disabled = !ready || busy || automatic || done;
  $('pause').disabled = !automatic;
  $('replay').disabled = busy || !run?.steps.length;
  $('step-count').innerHTML = `${String(run?.steps.length || 0).padStart(2, '0')} <small>/ ${run?.limit || 30} 步</small>`;
  $('total-time').textContent = run?.steps.length ? `${(run.steps.reduce((n,s) => n + s.elapsed_ms, 0) / 1000).toFixed(1)}s` : '—';
  $('last-move').textContent = run?.steps.at(-1)?.choice || '—';
  $('phase').textContent = labels[run?.status] || '准备就绪';
  $('phase').className = `pill ${run?.status === 'solved' ? 'success' : busy ? 'busy' : ''}`;
}
function persist() {
  if (run?.steps.length || run?.status === 'error') {
    archive = archive.filter(r => r.id !== run.id);
    // ponytail: keep only 100 local runs; export JSON for longer experiments.
    archive.push(structuredClone(run));
    archive = archive.slice(-100);
    try { localStorage.setItem(storageKey, JSON.stringify(archive)); }
    catch { message('浏览器无法保存记录，请使用导出 JSON 保存本次实验。', true); }
  }
  renderResults();
}
function renderResults() {
  $('results').replaceChildren();
  for (const r of [...archive].reverse()) {
    const row = document.createElement('tr');
    const status = ['running', 'paused', 'error'].includes(r.status) && r.id !== run?.id ? '未完成' : labels[r.status];
    for (const value of [new Date(r.started).toLocaleTimeString('zh-CN'), r.depth, status, r.steps.length,
      `${(r.steps.reduce((n,s)=>n+s.elapsed_ms,0)/1000).toFixed(1)}s`, r.steps.at(-1)?.model || '—']) {
      const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
      if (value === labels.solved) cell.className = 'solved';
    }
    $('results').append(row);
  }
  const completed = archive.filter(r => ['solved','cycle','limit'].includes(r.status));
  $('stats').textContent = completed.length ? [...new Set(completed.map(r => r.depth))].sort((a,b)=>a-b).map(depth => {
    const group = completed.filter(r=>r.depth === depth), solved = group.filter(r=>r.status === 'solved').length;
    return `${depth} 步打乱：${solved}/${group.length} 成功（${Math.round(solved/group.length*100)}%）`;
  }).join('　·　') : '还没有完成的实验。暂停、请求失败和提前结束不计入成功率。';
}
function renderDecision(step) {
  $('choice').textContent = step?.choice || '—';
  $('confidence').textContent = step ? `${Math.round(step.confidence * 100)}%` : '—';
  $('latency').textContent = step ? `${(step.elapsed_ms/1000).toFixed(2)}s / 判断` : '等待判断';
  if (step) $('model').textContent = step.model;
  $('input').textContent = step ? JSON.stringify(step.input, null, 2) : '执行第一步后显示。';
  $('probabilities').replaceChildren();
  for (const move of MOVES) {
    const value = step?.probabilities[move];
    const el = document.createElement('div');
    el.className = `probability ${step?.choice === move ? 'selected' : ''}`;
    if (step && value == null) el.title = step.input.repeated_moves?.includes(move) ? '执行后会回到本轮已出现状态' : '上一步动作的逆操作';
    const label = document.createElement('div'); label.className = 'prob-label';
    const name = document.createElement('span'); name.textContent = move;
    const number = document.createElement('span'); number.textContent = value == null ? (step ? '禁选' : '—') : `${(value*100).toFixed(1)}%`;
    label.append(name, number);
    const track = document.createElement('div'); track.className = 'track';
    const bar = document.createElement('div'); bar.className = 'bar'; bar.style.width = `${(value || 0)*100}%`;
    track.append(bar); el.append(label, track); $('probabilities').append(el);
  }
}
function renderHistory(active = run.steps.length - 1) {
  $('history-count').textContent = run.steps.length;
  $('timeline').replaceChildren();
  if (!run.steps.length) {
    const p = document.createElement('p'); p.className = 'muted'; p.textContent = '每一个选择，都会留在这里。'; $('timeline').append(p);
  }
  run.steps.forEach((step, index) => {
    const button = document.createElement('button'); button.className = `move-chip ${active === index ? 'active' : ''}`;
    button.disabled = busy;
    const num = document.createElement('small'); num.textContent = String(index+1).padStart(2,'0');
    const move = document.createElement('b'); move.textContent = step.choice; button.append(num, move);
    button.setAttribute('aria-label', `回看第 ${index+1} 步 ${step.choice}`);
    button.onclick = () => {
      player.alg = run.steps.slice(0,index+1).map(s=>s.choice).join(' '); player.timestamp = 'end';
      renderDecision(step); renderHistory(index);
      message(`回看第 ${index+1} 步：左侧为执行后状态，右侧为该步决策。继续执行时恢复当前状态。`);
    };
    $('timeline').append(button);
  });
}
async function animate(moves) {
  player.alg = moves.slice(0,-1).join(' ');
  const start = (await player.experimentalModel.timeRange.get()).end;
  player.timestamp = start;
  player.alg = moves.join(' ');
  const end = (await player.experimentalModel.timeRange.get()).end;
  const duration = matchMedia('(prefers-reduced-motion: reduce)').matches ? 60 : 650;
  await new Promise(resolve => {
    const begin = performance.now();
    const frame = now => {
      const progress = Math.min((now - begin) / duration, 1);
      player.timestamp = start + (end - start) * progress;
      if (progress < 1) requestAnimationFrame(frame); else resolve();
    };
    requestAnimationFrame(frame);
  });
}
function newRun() {
  if (busy) return;
  if (run?.steps.length && !terminal.has(run.status)) { run.status = 'abandoned'; persist(); }
  automatic = false;
  const depth = Number($('depth').value), mix = scramble(depth);
  cube = new Cube().move(mix.join(' '));
  run = {id:crypto.getRandomValues(new Uint32Array(4)).join('-'), started:new Date().toISOString(), depth, limit:Number($('limit').value), scramble:mix, initial:faceState(cube), status:'ready', steps:[]};
  player.experimentalSetupAlg = mix.join(' '); player.alg = ''; player.timestamp = 0;
  renderDecision(null); renderHistory(); controls(); renderResults();
  message(ready ? `已随机打乱 ${depth} 步。Jev 将从当前状态开始判断。` : '请在服务器配置 TYPESAFE_API_KEY 后重启并刷新。', !ready);
}
async function requestStep(input) {
  const started = performance.now();
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      const response = await fetch('/api/step', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(input), signal:AbortSignal.timeout(55000)});
      const answer = await response.json();
      if (!response.ok) {
        const error = new Error(answer.error || '请求失败，本步未执行');
        error.retryable = answer.retryable === true;
        throw error;
      }
      return {...answer, attempts:attempt, elapsed_ms:Math.round(performance.now() - started)};
    } catch (error) {
      const retryable = error.retryable || error.name === 'TimeoutError' || error instanceof TypeError;
      if (!retryable || attempt === 3 || pauseRequested) throw error;
      message(`连接暂时失败，${attempt} 秒后重试（第 ${attempt+1}/3 次），魔方状态保持不变。`);
      await new Promise(resolve => setTimeout(resolve, attempt * 1000));
      if (pauseRequested) throw new Error('已暂停重试，本步未执行。');
    }
  }
}
async function step() {
  if (busy || !ready || terminal.has(run.status)) return;
  busy = true; pauseRequested = false; run.status = 'running'; controls(); renderHistory();
  player.alg = run.steps.map(s=>s.choice).join(' '); player.timestamp = 'end';
  message(`Jev 正在判断第 ${run.steps.length+1} 步…`);
  const input = decisionInput(cube, run.steps);
  try {
    const answer = await requestStep(input);
    const record = {...answer, input};
    renderDecision(record);
    message(`Jev 选择 ${answer.choice}，正在执行…`);
    await animate([...input.history, answer.choice]);
    cube.move(answer.choice);
    run.steps.push({...record, after:faceState(cube)});
    const end = outcome(cube, run.steps.length, run.limit);
    run.status = end || (automatic ? 'running' : 'paused');
    if (end) automatic = false;
    message(end === 'solved' ? `还原成功！Jev 共执行 ${run.steps.length} 步。` :
      end === 'limit' ? `已达到 ${run.limit} 步上限，尚未还原。` :
      automatic ? '动作已完成，准备下一次判断。' : '本步已完成，可以继续或回看。');
    persist();
  } catch (error) {
    automatic = false; run.status = pauseRequested ? 'paused' : 'error';
    message(error.name === 'TimeoutError' ? '请求超时，本步未执行，可以重试。' : error.message, true);
    persist();
  } finally { busy = false; controls(); renderHistory(); }
}
$('step').onclick = () => step();
$('auto').onclick = async () => { automatic = true; controls(); while (automatic) await step(); };
$('pause').onclick = () => { automatic = false; pauseRequested = true; message('将在当前动作完成后暂停。'); controls(); };
$('new').onclick = newRun;
$('depth').onchange = newRun;
$('limit').onchange = () => { if (run && !run.steps.length) {run.limit = Number($('limit').value); controls();} else newRun(); };
$('replay').onclick = async () => {
  if (busy || !run.steps.length) return;
  busy = true; controls(); renderHistory();
  try {
    for (let i=0;i<run.steps.length;i++) {
      message(`重放 ${i+1} / ${run.steps.length}，不调用 Jev。`);
      renderDecision(run.steps[i]); renderHistory(i);
      await animate(run.steps.slice(0,i+1).map(s=>s.choice));
    }
    message('重放完成。');
  } catch { message('动画重放失败，请刷新页面。', true); }
  finally { busy = false; controls(); renderHistory(); }
};
$('export').onclick = () => {
  persist();
  const blob = new Blob([JSON.stringify(archive,null,2)], {type:'application/json'});
  const url = URL.createObjectURL(blob), a = document.createElement('a');
  a.href = url; a.download = `jev-cube-${new Date().toISOString().slice(0,10)}.json`; a.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
};
async function init() {
  try {
    player = new TwistyPlayer({puzzle:'3x3x3', background:'none', controlPanel:'none', hintFacelets:'none', viewerLink:'none', cameraLatitude:26, cameraLongitude:30, experimentalDragInput:'none'});
    $('cube').append(player);
    const config = await (await fetch('/api/config')).json();
    ready = config.ready;
    $('model').textContent = config.model;
    $('connection').textContent = ready ? '密钥已配置' : '尚未配置密钥';
    $('connection').classList.toggle('ready',ready);
    newRun();
  } catch(error) { message(`加载失败：${error.message}`,true); }
}
init();
