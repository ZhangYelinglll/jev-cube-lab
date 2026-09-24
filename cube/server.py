"""Local Jev cube lab. Run: uv run --env-file .env python -m cube.server"""
import argparse
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mimetypes
import os
from pathlib import Path
import time
from urllib.parse import urlsplit

import httpx

ROOT = Path(__file__).resolve().parents[1] / 'web'
MOVES = [f + s for f in 'URFDLB' for s in ('', "'", '2')]
FACE_NAMES = dict(zip('URFDLB', ('up', 'right', 'front', 'down', 'left', 'back')))


def validate_snapshot(value):
    if not isinstance(value, str) or len(value) != 54 or Counter(value) != Counter({f:9 for f in FACE_NAMES}):
        raise ValueError('轨迹状态无效')
    if any(value[i * 9 + 4] != f for i, f in enumerate(FACE_NAMES)):
        raise ValueError('轨迹中心块无效')
    return value


def make_payload(data):
    if not isinstance(data, dict):
        raise ValueError('状态必须是对象')
    faces, history = data.get('faces'), data.get('history')
    if not isinstance(faces, dict) or set(faces) != set('URFDLB'):
        raise ValueError('需要六个面的状态')
    if any(not isinstance(v, list) or len(v) != 9 or
           any(not isinstance(c, str) or c not in FACE_NAMES for c in v) for v in faces.values()):
        raise ValueError('每个面需要九个合法面片')
    if Counter(c for v in faces.values() for c in v) != Counter({f: 9 for f in FACE_NAMES}):
        raise ValueError('每种面片必须恰好九个')
    if any(faces[f][4] != f for f in FACE_NAMES):
        raise ValueError('中心块必须固定')
    if not isinstance(history, list) or len(history) >= 50 or any(m not in MOVES for m in history):
        raise ValueError('动作历史无效或已达到 50 步上限')
    last = history[-1] if history else None
    inverse = (last if last.endswith('2') else last[0] if last.endswith("'") else last + "'") if last else None
    repeated = data.get('repeated_moves', [])
    if not isinstance(repeated, list) or len(repeated) > 18 or any(m not in MOVES for m in repeated):
        raise ValueError('重复状态候选无效')
    available = [m for m in MOVES if m != inverse]
    unvisited = [m for m in available if m not in repeated]
    allowed = unvisited or available
    trajectory = data.get('trajectory')
    if trajectory is None:
        if history:
            raise ValueError('缺少动作轨迹，请刷新页面')
        trajectory = {'initial_faces': ''.join(''.join(faces[f]) for f in FACE_NAMES), 'steps': []}
    if not isinstance(trajectory, dict) or not isinstance(trajectory.get('steps'), list) or len(trajectory['steps']) != len(history):
        raise ValueError('轨迹与历史长度不一致')
    initial = validate_snapshot(trajectory.get('initial_faces'))
    steps = []
    for move, entry in zip(history, trajectory['steps']):
        if not isinstance(entry, dict) or entry.get('move') != move:
            raise ValueError('轨迹动作不一致')
        steps.append({'move': move, 'faces_after': validate_snapshot(entry.get('faces_after'))})
    if (steps[-1]['faces_after'] if steps else initial) != ''.join(''.join(faces[f]) for f in FACE_NAMES):
        raise ValueError('轨迹末尾与当前状态不一致')
    return {
        'model': os.getenv('TYPESAFE_MODEL', 'jev-1.13.0'),
        'state': {
            'puzzle': '3x3x3 Rubiks cube', 'faces': faces, 'your_previous_moves': history,
            'trajectory': {'initial_faces': initial, 'steps': steps},
            'repeat_filter_relaxed': not bool(unvisited),
            'encoding': 'Each sticker letter identifies its home face / center color. '
                        'U=up, R=right, F=front, D=down, L=left, B=back. '
                        'Each face is row-major viewed from outside. For F,R,B,L the top edge '
                        'touches U. For U the top edge touches B; for D the top edge touches F. '
                        'Centers and the coordinate frame stay fixed. Camera motion is irrelevant. '
                        'Trajectory snapshots are 54-letter strings: nine stickers per face in '
                        'U,R,F,D,L,B order, with the same row-major layout. initial_faces is the '
                        'starting scrambled state, not the scramble moves. Each step records your '
                        'move and the resulting faces_after state.',
            'goal': 'Every sticker on each face must match that face center. Choose the next '
                    'single face turn toward solving the entire cube. You may need intermediate '
                    'moves that temporarily disturb solved stickers.',
        },
        'questions': {'move': {
            'type': 'choice',
            'instructions': 'Select the next move to solve the cube from its current faces. '
                            'Use standard Singmaster notation, viewed directly at the turning '
                            'face from outside. Consecutive turns on the same face are allowed. '
                            'Only the immediate inverse is always excluded. The simulator also '
                            'excludes moves that revisit a recorded state, unless all available '
                            'moves do so (repeat_filter_relaxed=true). '
                            'Two consecutive quarter turns can instead be one half turn (U U = U2); '
                            'four identical quarter turns do nothing. Review trajectory '
                            'as background experience: compare current faces with earlier snapshots '
                            'and learn from moves that returned to a prior state. Repeated states '
                            'do not stop the experiment; choose a useful next move.',
            'criteria': {m: f"Turn the {FACE_NAMES[m[0]]} face " +
                         ('180 degrees' if m.endswith('2') else
                          '90 degrees counterclockwise' if m.endswith("'") else '90 degrees clockwise')
                         for m in allowed},
        }},
    }


def validate_answer(result, allowed_moves=MOVES):
    try:
        answer = result['answers']['move']
        probs = answer['probabilities']
        if answer['type'] != 'choice' or answer['choice'] not in allowed_moves or set(probs) != set(allowed_moves):
            raise ValueError()
        for number in [answer['confidence'], *probs.values()]:
            if isinstance(number, bool) or not isinstance(number, (float, int)) or not math.isfinite(number) or not 0 <= number <= 1:
                raise ValueError()
        if not math.isclose(sum(probs.values()), 1, abs_tol=0.002) or not isinstance(result['model'], str):
            raise ValueError()
        return {**answer, 'model': result['model'], 'usage': result.get('usage', {})}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Jev 返回了无效动作或概率，本步未执行') from exc


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, value):
        body = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == '/api/config':
            return self.reply(200, {'ready': bool(os.getenv('TYPESAFE_API_KEY')),
                                    'model': os.getenv('TYPESAFE_MODEL', 'jev-1.13.0')})
        # Only serve this app, never the workspace or .env.
        file = (ROOT / ('index.html' if path == '/' else path.lstrip('/'))).resolve()
        if not file.is_relative_to(ROOT.resolve()) or not (
            file == ROOT / 'index.html' or file == ROOT / 'style.css' or file.is_relative_to(ROOT / 'dist')
        ) or not file.is_file():
            return self.reply(404, {'error': 'Not found'})
        body = file.read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', mimetypes.guess_type(file.name)[0] or 'application/octet-stream')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != '/api/step':
            return self.reply(404, {'error': 'Not found'})
        origin = self.headers.get('Origin')
        if origin and urlsplit(origin).netloc != self.headers.get('Host'):
            return self.reply(403, {'error': '不允许跨站请求'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 16384:
                raise ValueError('请求大小无效')
            payload = make_payload(json.loads(self.rfile.read(length)))
        except (ValueError, TypeError):
            return self.reply(400, {'error': '魔方状态或动作历史无效'})
        key = os.getenv('TYPESAFE_API_KEY')
        if not key:
            return self.reply(503, {'error': '请在服务器 .env 中配置 TYPESAFE_API_KEY 并重启'})
        try:
            started = time.monotonic()
            response = httpx.post('https://api.typesafe.ai/v1/systemone', json=payload,
                                  headers={'Authorization': 'Bearer ' + key}, timeout=45)
            response.raise_for_status()
            answer = validate_answer(response.json(), payload['questions']['move']['criteria'])
            answer['elapsed_ms'] = round((time.monotonic() - started) * 1000)
            self.reply(200, answer)
        except httpx.HTTPStatusError as exc:
            self.reply(502, {'error': f'Jev 服务返回 HTTP {exc.response.status_code}，本步未执行'})
        except httpx.RequestError as exc:
            # Log exception classes, never proxy URLs, request headers or API keys.
            self.log_message('Jev %s cause=%s elapsed=%.2fs history=%d',
                             type(exc).__name__, type(exc.__cause__).__name__,
                             time.monotonic() - started, len(payload['state']['your_previous_moves']))
            self.reply(502, {'error': f'连接 Jev 失败（{type(exc).__name__}），本步未执行，可以重试',
                             'retryable': True})
        except ValueError as exc:
            self.reply(502, {'error': str(exc)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--host', default='127.0.0.1')
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'Jev Cube Lab: http://{args.host}:{server.server_port}', flush=True)
    server.serve_forever()
