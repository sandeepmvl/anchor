import asyncio
import contextlib
import html
import json
import os
import re
import socket
import struct
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import requests
import uvicorn

from anchor_ai.app import app
from anchor_ai import observability, streaming
from anchor_ai.llm_client import LlmClient


ROOT = Path(__file__).resolve().parents[2]


def port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def eventually(check, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(0.02)
    raise AssertionError('timed out waiting for stack evidence')


def test_browser_disconnect_cancels_upstream_provider(monkeypatch):
    python_port, rails_port = port(), port()
    cancelled, closed = threading.Event(), threading.Event()
    records = []
    slow = threading.Event()

    async def source(message):
        try:
            yield streaming.Delta(text='Refunds ')
            try:
                await asyncio.sleep(60 if slow.is_set() else 0.15)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            yield streaming.Delta(text='take five days.')
            yield streaming.Delta(stop='stop')
        finally:
            closed.set()

    async def client_source(self, prompt, *, contract=None):
        upstream = source(prompt)
        try:
            async for delta in upstream:
                if delta.text:
                    yield delta.text
        finally:
            await upstream.aclose()

    monkeypatch.setenv('ANCHOR_AI_SECRET', 'stack-only-secret')
    monkeypatch.setattr(streaming, 'TOKEN_SOURCE', source)
    monkeypatch.setattr(LlmClient, 'stream_tokens', client_source)
    monkeypatch.setattr(observability, 'SINK', records.append)
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=python_port,
                                         log_level='error', lifespan='off'))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    process = None
    browser = None
    try:
        eventually(lambda: server.started)
        env = dict(os.environ, RAILS_ENV='test', ANCHOR_AI_SECRET='stack-only-secret',
                   ANCHOR_AI_URL=f'http://127.0.0.1:{python_port}/v1/infer',
                   ANCHOR_AI_OPEN_TIMEOUT='1', ANCHOR_AI_READ_TIMEOUT='2')
        with tempfile.TemporaryFile(mode='w+b') as logs:
            process = subprocess.Popen(
                ['ruby', 'bin/rails', 'server', '-b', '127.0.0.1', '-p', str(rails_port)],
                cwd=ROOT, env=env, stdout=logs, stderr=logs)
            base = f'http://127.0.0.1:{rails_port}'

            def ready():
                if process.poll() is not None:
                    logs.seek(0)
                    raise AssertionError(logs.read().decode(errors='replace'))
                try:
                    return requests.get(base + '/up', timeout=0.2).status_code == 200
                except requests.RequestException:
                    return False

            eventually(ready)
            session = requests.Session()
            page = session.get(base + '/stream_chat', timeout=2)
            page.raise_for_status()
            match = re.search(r'name="csrf-token" content="([^"]+)"', page.text)
            token = html.unescape(match[1]) if match else ''
            headers = {'Accept': 'text/event-stream', 'X-CSRF-Token': token}
            with session.post(base + '/chat', json={'message': 'When will my refund arrive?', 'stream': True},
                              headers=headers, stream=True, timeout=2) as response:
                response.raise_for_status()
                packets, buffer = [], b''
                for chunk in response.iter_content(chunk_size=1):
                    buffer += chunk
                    if buffer.endswith(b'\n\n'):
                        packets.append(buffer)
                        buffer = b''
                tokens = [p for p in packets if p.startswith(b'event: token')]
                assert len(tokens) == 2
                assert packets.index(tokens[0]) < packets.index(tokens[1])
                assert packets[-1].startswith(b'event: final')
                assert b'Refunds take five days.' in packets[-1]
            closed.clear()
            slow.set()
            body = json.dumps({'message': 'When will my refund arrive?', 'stream': True}).encode()
            cookie = '; '.join(f'{key}={value}' for key, value in session.cookies.items())
            browser = socket.create_connection(('127.0.0.1', rails_port), timeout=2)
            request = (
                f'POST /chat HTTP/1.1\r\nHost: 127.0.0.1:{rails_port}\r\n'
                f'Content-Type: application/json\r\nAccept: text/event-stream\r\n'
                f'Cookie: {cookie}\r\nX-CSRF-Token: {token}\r\n'
                f'Content-Length: {len(body)}\r\n\r\n').encode() + body
            browser.sendall(request)
            received = b''
            while b'event: token' not in received:
                received += browser.recv(4096)
            browser.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))
            browser.close()
            browser = None
            eventually(cancelled.is_set)
            eventually(closed.is_set)
            eventually(lambda: any(r['name'] == 'python.stream' and r['status'] == 'error'
                                   for r in records))
            eventually(lambda: (logs.seek(0) is not None and
                       b'rails.infer_call' in logs.read()))
            print('Anchor: the same answer appears progressively; closing the connection stops remaining generation.')
    finally:
        if browser is not None:
            browser.close()
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        server.should_exit = True
        thread.join(timeout=3)
        assert not thread.is_alive(), 'temporary Python server did not stop'
