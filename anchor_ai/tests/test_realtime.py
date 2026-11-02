import asyncio
import importlib.util
import json
import struct
import threading

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from anchor_ai.realtime import Interaction, fixture_app, register, FixturePorts, PCM, fixture_loader, Ledger
from anchor_ai.escalation import redact

SCOPE = {'tenant_id': 1, 'user_id': 3, 'ticket_id': 11}


def receive(ws, *, raw=False):
    timer = threading.Timer(2, ws.close)
    timer.daemon = True
    timer.start()
    try:
        return ws.receive() if raw else ws.receive_json()
    finally:
        timer.cancel()


def test_complete_interruptible_voice_flow(tmp_path):
    spec = importlib.util.spec_from_file_location('voice_client', 'labs/realtime-voice/voice_client.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.exercise(tmp_path)
    assert result['side_effects'] == 0
    assert result['bounded_queue']
    for measurements in result['stage_ms'].values():
        assert 0 <= measurements['stt_ms'] <= 200
        assert 0 <= measurements['generation_ms'] <= 300
        assert 0 <= measurements['tts_ms'] <= 200


def test_credential_origin_expiry_replay_and_tenant_reservations(tmp_path):
    ledger = Ledger(tmp_path / 'ledger', sessions=1, units=20)
    token = ledger.test_issue(SCOPE)
    with pytest.raises(PermissionError):
        ledger.redeem(token, 'https://evil.invalid', 'bad')
    assert ledger.redeem(token, 'http://testserver', 'a') == SCOPE
    with pytest.raises(PermissionError):
        ledger.redeem(token, 'http://testserver', 'replay')
    rejected = ledger.test_issue(SCOPE)
    with pytest.raises(OverflowError):
        ledger.redeem(rejected, 'http://testserver', 'b')
    with pytest.raises(PermissionError):
        ledger.redeem(rejected, 'http://testserver', 'retry')
    other = dict(SCOPE, tenant_id=2)
    assert ledger.redeem(ledger.test_issue(other), 'http://testserver', 'c') == other
    ledger.finish('a', SCOPE, {})
    ledger.redeem(ledger.test_issue(SCOPE), 'http://testserver', 'd')
    ledger.finish('d', SCOPE, {})
    with pytest.raises(OverflowError):
        ledger.redeem(ledger.test_issue(SCOPE), 'http://testserver', 'spent')
    with pytest.raises(PermissionError):
        ledger.redeem(ledger.test_issue(other, ttl=-1), 'http://testserver', 'expired')


def test_media_before_authentication_and_client_authority_are_rejected(tmp_path):
    app = fixture_app(tmp_path / 'db', tmp_path / 'kill')
    with TestClient(app) as client:
        with client.websocket_connect('/v1/realtime') as ws:
            ws.send_bytes(PCM)
            assert receive(ws, raw=True)['code'] == 1008
        token = Ledger(tmp_path / 'db').test_issue(SCOPE)
        with client.websocket_connect('/v1/realtime', headers={'origin': 'http://testserver'}) as ws:
            ws.send_json({'type': 'authenticate', 'credential': token, 'tenant_id': 2})
            assert receive(ws, raw=True)['code'] == 1008


def test_owned_screenshot_only():
    assert fixture_loader(SCOPE, 7)[0]['untrusted'] is True
    with pytest.raises(PermissionError):
        fixture_loader(dict(SCOPE, tenant_id=2), 7)


def test_human_only_stops_generation_and_persists_handoff(tmp_path):
    ledger = Ledger(tmp_path / 'db')
    kill = tmp_path / 'kill'
    app = fixture_app(tmp_path / 'db', kill)
    with TestClient(app) as client:
        with client.websocket_connect('/v1/realtime', headers={'origin': 'http://testserver'}) as ws:
            ws.send_json({'type': 'authenticate', 'credential': ledger.test_issue(SCOPE)})
            assert receive(ws)['type'] == 'authenticated'
            kill.touch()
            assert receive(ws, raw=True)['type'] == 'websocket.close'
    payload = ledger.handoffs()[0]
    assert payload['reason'] == 'human_only'
    assert payload['pending_action'] == 'none_executed'
    assert payload['approval_required'] is True
    assert 'person@example.com' not in json.dumps(redact({'transcript': ['person@example.com']}))


def test_sequence_gap_is_observable_and_fails_closed(tmp_path):
    ledger = Ledger(tmp_path / 'db')
    app = fixture_app(tmp_path / 'db', tmp_path / 'kill')
    with TestClient(app) as client:
        with client.websocket_connect('/v1/realtime', headers={'origin': 'http://testserver'}) as ws:
            ws.send_json({'type': 'authenticate', 'credential': ledger.test_issue(SCOPE)})
            receive(ws)
            ws.send_bytes(struct.pack('!Q', 3) + PCM)
            assert receive(ws, raw=True)['type'] == 'websocket.close'
    assert any(row['event'] == 'frame_gap' for row in app.state.realtime_traces[0])


def test_slow_send_has_a_deadline_and_queue_is_bounded(tmp_path):
    class SlowSocket:
        async def send_json(self, item):
            await asyncio.sleep(1)
    async def exercise():
        instance = Interaction(SlowSocket(), SCOPE, 'trace', Ledger(tmp_path / 'db'),
            FixturePorts(), fixture_loader, tmp_path / 'kill', [], redact)
        assert instance.queue.maxsize == 4
        with pytest.raises(TimeoutError):
            await instance.direct({'type': 'test'})
    asyncio.run(exercise())


def test_stage_timeout_cancels_and_hands_off(tmp_path):
    class SlowPorts(FixturePorts):
        async def transcribe(self, pcm):
            await asyncio.sleep(1)
            return 'yes, approve the refund'
    ledger = Ledger(tmp_path / 'db')
    app = register(FastAPI(), ledger=ledger, ports=SlowPorts(), loader=fixture_loader,
                   handoff=redact, kill_file=tmp_path / 'kill')
    with TestClient(app) as client:
        with client.websocket_connect('/v1/realtime', headers={'origin': 'http://testserver'}) as ws:
            ws.send_json({'type': 'authenticate', 'credential': ledger.test_issue(SCOPE)})
            receive(ws)
            for number, pcm in enumerate([PCM] + [bytes(640)] * 10):
                ws.send_bytes(struct.pack('!Q', number) + pcm)
            assert receive(ws)['type'] == 'flush'
            assert receive(ws, raw=True)['type'] == 'websocket.close'
    assert ledger.handoffs()[0]['pending_action'] == 'none_executed'
    assert ledger.handoffs()[0]['reason'] == 'policy_or_stage_failure'


def test_production_gate_rejects_missing_evidence():
    from eval.agent_gate import require_release, require_release_path
    with pytest.raises(ValueError, match='realtime_release_evidence_missing'):
        require_release(None)
    with pytest.raises(ValueError, match='realtime_release_evidence_incomplete'):
        require_release({'offline_fixture': True})
    with pytest.raises(ValueError, match='realtime_release_evidence_missing'):
        require_release_path('')


def test_failed_task_does_not_abort_sibling_cleanup():
    from anchor_ai.streaming import cancel_tasks
    async def exercise():
        stopped = asyncio.Event()
        async def failed():
            raise ValueError('stage_failed')
        async def sibling():
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
        bad = asyncio.create_task(failed())
        waiting = asyncio.create_task(sibling())
        await asyncio.sleep(0)
        await asyncio.wait_for(cancel_tasks([bad, waiting, None]), 0.1)
        assert bad.done() and waiting.done() and stopped.is_set()
        assert isinstance(bad.exception(), ValueError)
    asyncio.run(exercise())
