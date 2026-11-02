import json
import struct
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fastapi.testclient import TestClient
from anchor_ai.realtime import fixture_app, PCM, Ledger

SCOPE = {'tenant_id': 1, 'user_id': 3, 'ticket_id': 11}


def send_turn(ws, start):
    for offset, pcm in enumerate([PCM] + [bytes(640)] * 10):
        ws.send_bytes(struct.pack('!Q', start + offset) + pcm)
    return start + 11


def bounded_receive(ws):
    """A failed server must fail the client, not hang the verifier indefinitely."""
    timer = threading.Timer(2, ws.close)
    timer.daemon = True
    timer.start()
    try:
        return ws.receive_json()
    finally:
        timer.cancel()


def exercise(directory):
    directory = Path(directory)
    path, kill = directory / 'ledger.sqlite3', directory / 'human-only'
    ledger = Ledger(path)
    app = fixture_app(path, kill)
    token = ledger.test_issue(SCOPE)
    received = []
    with TestClient(app) as client:
        with client.websocket_connect('/v1/realtime', headers={'origin': 'http://testserver'}) as ws:
            ws.send_json({'type': 'authenticate', 'credential': token})
            assert bounded_receive(ws)['type'] == 'authenticated'
            ws.send_json({'type': 'attachment', 'attachment_id': 7})
            assert bounded_receive(ws)['type'] == 'context_ready'
            sequence = send_turn(ws, 0)
            assert bounded_receive(ws)['type'] == 'flush'
            while True:
                event = bounded_receive(ws)
                received.append(event)
                if event['type'] == 'sentence_end':
                    ws.send_json({'type': 'played', 'epoch': event['epoch'], 'id': event['id']})
                    if event['id'] == 1:
                        break
            send_turn(ws, sequence)
            while True:
                event = bounded_receive(ws)
                received.append(event)
                if event['type'] == 'flush':
                    assert len(event['kept']) == 1
                    epoch = event['epoch']
                    break
            while True:
                event = bounded_receive(ws)
                received.append(event)
                assert event['epoch'] == epoch
                if event['type'] == 'reply_done':
                    break
    assert all(row['pending_action'] == 'none_executed' for row in ledger.handoffs())
    assert any(row['type'] == 'transcript' for row in received)
    assert any(row['type'] == 'audio' for row in received)
    assert any(row.get('citations') for row in received)
    assert [row['sequence'] for row in received] == sorted(row['sequence'] for row in received)
    trace = app.state.realtime_traces[0]
    assert max(row['queue_depth'] for row in trace) <= 4
    stages = {}
    for epoch in (1, 2):
        rows = [row for row in trace if row['epoch'] == epoch]
        def first(name):
            return next(row['t'] for row in rows if row['event'] == name)
        stages[str(epoch)] = {
            'stt_ms': (first('stt_done') - first('stt_start')) * 1000,
            'generation_ms': (first('answer_text') - first('generation_start')) * 1000,
            'tts_ms': (first('audio') - first('tts_start')) * 1000}
    return {'demo': 'The same recorded question and screenshot now drive an interruptible realtime conversation with incremental audio responses instead of waiting for a completed attachment-processing round trip.',
            'ordered_events': [row['type'] for row in received],
            'stage_ms': stages, 'bounded_queue': True, 'side_effects': 0,
            'not_verified': ['real speech quality', 'network latency', 'audible playback',
                             'distributed quotas', 'Rails authorization adapter']}


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as directory:
        print(json.dumps(exercise(directory), indent=2))
