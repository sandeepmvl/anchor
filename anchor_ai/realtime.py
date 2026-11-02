import asyncio
import base64
import binascii
import contextlib
import hashlib
import json
import math
import os
import sqlite3
import struct
import time
import uuid
import zlib
from contextlib import contextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket

SCHEMA = '''
CREATE TABLE IF NOT EXISTS realtime_credentials (
 digest TEXT PRIMARY KEY, scope TEXT NOT NULL, origin TEXT NOT NULL,
 expires REAL NOT NULL, used INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS realtime_reservations (
 session TEXT PRIMARY KEY, tenant INTEGER NOT NULL, expires REAL NOT NULL);
CREATE TABLE IF NOT EXISTS realtime_spending (
 tenant INTEGER NOT NULL, window INTEGER NOT NULL, units INTEGER NOT NULL,
 starts INTEGER NOT NULL, PRIMARY KEY(tenant,window));
CREATE TABLE IF NOT EXISTS realtime_handoffs (
 session TEXT PRIMARY KEY, tenant INTEGER NOT NULL, ticket INTEGER NOT NULL,
 payload TEXT NOT NULL);
'''


class Ledger:
    """Single-host lab ledger, not a distributed quota implementation."""
    def __init__(self, path, *, sessions=2, units=40, starts=10):
        self.path = str(path)
        self.session_cap, self.unit_cap, self.start_cap = sessions, units, starts
        with sqlite3.connect(self.path) as db:
            db.executescript(SCHEMA)

    @contextmanager
    def transaction(self):
        db = sqlite3.connect(self.path, timeout=0.2)
        try:
            db.execute('BEGIN IMMEDIATE')
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def redeem(self, token, origin, session):
        if not isinstance(token, str) or not 32 <= len(token) <= 128:
            raise PermissionError('credential_rejected')
        digest = hashlib.sha256(token.encode()).hexdigest()
        now = time.time()
        with self.transaction() as db:
            row = db.execute('SELECT scope,origin,expires,used FROM realtime_credentials '
                             'WHERE digest=?', (digest,)).fetchone()
            if not row or row[1] != origin or row[2] <= now or row[3]:
                raise PermissionError('credential_rejected')
            scope = json.loads(row[0])
            if set(scope) != {'tenant_id', 'user_id', 'ticket_id'} or any(
                    type(v) is not int or v <= 0 for v in scope.values()):
                raise PermissionError('credential_rejected')
            tenant = scope['tenant_id']
            db.execute('UPDATE realtime_credentials SET used=1 WHERE digest=?', (digest,))
            db.execute('DELETE FROM realtime_reservations WHERE expires<=?', (now,))
            count = db.execute('SELECT COUNT(*) FROM realtime_reservations WHERE tenant=?',
                               (tenant,)).fetchone()[0]
            window = int(now // 60)
            db.execute('INSERT OR IGNORE INTO realtime_spending VALUES(?,?,0,0)', (tenant, window))
            units, starts = db.execute('SELECT units,starts FROM realtime_spending '
                                      'WHERE tenant=? AND window=?', (tenant, window)).fetchone()
            admitted = (count < self.session_cap and units + 10 <= self.unit_cap
                        and starts < self.start_cap)
            if admitted:
                db.execute('INSERT INTO realtime_reservations VALUES(?,?,?)',
                           (session, tenant, now + 60))
                db.execute('UPDATE realtime_spending SET units=units+10,starts=starts+1 '
                           'WHERE tenant=? AND window=?', (tenant, window))
        if not admitted:
            raise OverflowError('tenant_reservation_rejected')
        return scope

    def finish(self, session, scope, payload):
        with self.transaction() as db:
            db.execute('INSERT OR REPLACE INTO realtime_handoffs VALUES(?,?,?,?)',
                       (session, scope['tenant_id'], scope['ticket_id'], json.dumps(payload)))
            db.execute('DELETE FROM realtime_reservations WHERE session=?', (session,))

    def handoffs(self):
        with sqlite3.connect(self.path) as db:
            return [json.loads(row[0]) for row in db.execute(
                'SELECT payload FROM realtime_handoffs ORDER BY session')]

    def test_issue(self, scope, origin='http://testserver', ttl=30):
        import secrets
        token = secrets.token_urlsafe(32)
        with self.transaction() as db:
            db.execute('INSERT INTO realtime_credentials VALUES(?,?,?,?,0)',
                       (hashlib.sha256(token.encode()).hexdigest(), json.dumps(scope),
                        origin, time.time() + ttl))
        return token


QUESTION = 'Can I get a refund for the R17 error?'
PCM = struct.pack('<320h', *[int(3000 * math.sin(i * 0.2)) for i in range(320)])


def screenshot():
    def chunk(kind, data):
        return (struct.pack('!I', len(data)) + kind + data
                + struct.pack('!I', binascii.crc32(kind + data) & 0xffffffff))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', 1, 1, 8, 2, 0, 0, 0))
            + chunk(b'tEXt', b'fixture\x00Refund dialog: R17')
            + chunk(b'IDAT', zlib.compress(b'\x00\xff\xff\xff')) + chunk(b'IEND', b''))


class FixturePorts:
    offline = True

    async def transcribe(self, pcm):
        await asyncio.sleep(0.001)
        if pcm != PCM:
            raise ValueError('unrecorded_audio_fixture')
        return QUESTION

    async def answer(self, question, evidence):
        await asyncio.sleep(0.001)
        yield ('Screenshot R17 is linked to this question.' if evidence
               else 'Screenshot processing is pending.')
        await asyncio.sleep(0.001)
        yield 'Refund eligibility needs the existing approval workflow.'

    async def synthesize(self, text):
        await asyncio.sleep(0.001)
        yield PCM
        await asyncio.sleep(0.001)
        yield PCM


def fixture_loader(scope, attachment_id):
    if (scope['tenant_id'], scope['ticket_id'], attachment_id) != (1, 11, 7):
        raise PermissionError('attachment_not_owned')
    return [{'attachment_id': 7, 'page': 1, 'region': 'refund_dialog',
             'tenant_id': 1, 'ticket_id': 11, 'span': 'Refund dialog: error R17',
             'untrusted': True, 'digest': hashlib.sha256(screenshot()).hexdigest()}]


def injection_cases():
    return [json.loads(line) for line in Path(
        'eval/datasets/multimodal_injection_cases.jsonl').read_text().splitlines() if line]


async def capture_injection(case):
    evidence = fixture_loader({'tenant_id': 1, 'ticket_id': 11, 'user_id': 3}, 7)
    question = QUESTION
    if case['source'] == 'transcript':
        question += ' ' + case['instruction']
    else:
        evidence[0]['span'] += ' ' + case['instruction']
    text = [part async for part in FixturePorts().answer(question, evidence)]
    return {'text': text, 'tool_calls': [], 'approval_granted': False,
            'evidence_ids': [row['attachment_id'] for row in evidence], 'offline_fixture': True}


async def cancel(task):
    if task is not None:
        task.cancel()
        # Retrieving a failed task's exception must not abort cleanup.
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task


class Interaction:
    def __init__(self, ws, scope, session, ledger, ports, loader, kill_file, trace, handoff):
        self.ws, self.scope, self.session, self.ledger = ws, scope, session, ledger
        self.ports, self.loader, self.kill_file = ports, loader, Path(kill_file)
        self.trace, self.handoff = trace, handoff
        self.queue = asyncio.Queue(maxsize=4)
        self.lock = asyncio.Lock()
        self.reply = None
        self.epoch, self.sequence, self.input_sequence = 0, 0, 0
        self.pcm, self.silence = bytearray(), 0
        self.evidence, self.messages, self.heard = [], [], []
        self.sentences, self.completed = [], set()
        self.ack, self.turns = 0, 0
        self.reason = 'disconnect'
        self.started = time.monotonic()

    def stamp(self, event, **fields):
        from anchor_ai.observability import realtime_event
        record = realtime_event(self.session, self.scope, event, self.epoch,
                                self.queue.qsize(), **fields)
        self.trace.append(record)

    async def direct(self, payload):
        async with self.lock:
            self.sequence += 1
            started = time.monotonic()
            await asyncio.wait_for(self.ws.send_json(dict(payload, sequence=self.sequence)), 0.1)
            self.stamp('send_done', send_ms=(time.monotonic() - started) * 1000)

    async def emit(self, kind, epoch, **fields):
        await asyncio.wait_for(self.queue.put(dict(type=kind, epoch=epoch, **fields)), 0.1)
        self.stamp('queued')

    async def writer(self):
        while True:
            item = await self.queue.get()
            try:
                async with self.lock:
                    if item['epoch'] != self.epoch:
                        continue
                    self.sequence += 1
                    started = time.monotonic()
                    await asyncio.wait_for(self.ws.send_json(dict(item, sequence=self.sequence)), 0.1)
                    self.stamp('send_done', send_ms=(time.monotonic() - started) * 1000)
                    if item['type'] == 'sentence_end':
                        self.completed.add(item['id'])
            finally:
                self.queue.task_done()

    async def interrupt(self):
        self.epoch += 1
        await cancel(self.reply)
        self.reply = None
        while not self.queue.empty():
            self.queue.get_nowait()
            self.queue.task_done()
        kept = self.sentences[:self.ack]
        self.heard.extend(kept)
        self.sentences, self.completed, self.ack = [], set(), 0
        await self.direct(dict(type='flush', epoch=self.epoch, kept=kept))
        self.stamp('barge_in')

    async def response(self, pcm, evidence, epoch):
        from anchor_ai.contracts import RealtimeTranscript
        from anchor_ai.runtime import voice_authority
        self.stamp('stt_start')
        question = await asyncio.wait_for(self.ports.transcribe(pcm), 0.2)
        question = RealtimeTranscript(text=question).text
        assert voice_authority()['confirmed'] is False
        self.stamp('stt_done')
        self.messages.append(question)
        await self.emit('transcript', epoch, text=question, untrusted=True)
        self.stamp('generation_start')
        source = self.ports.answer(question, evidence).__aiter__()
        try:
            for sentence_id in range(1, 9):
                try:
                    text = await asyncio.wait_for(anext(source), 0.3)
                except StopAsyncIteration:
                    break
                if not isinstance(text, str) or not 0 < len(text) <= 1000:
                    raise ValueError('answer_chunk_rejected')
                self.sentences.append(text)
                self.stamp('answer_text')
                citations = [{k: row[k] for k in ('attachment_id', 'page', 'region')}
                             for row in evidence]
                await self.emit('answer_text', epoch, id=sentence_id, text=text, citations=citations)
                self.stamp('tts_start')
                audio_source = self.ports.synthesize(text).__aiter__()
                try:
                    for chunk_id in range(1, 51):
                        try:
                            audio = await asyncio.wait_for(anext(audio_source), 0.2)
                        except StopAsyncIteration:
                            break
                        if not isinstance(audio, bytes) or not 0 < len(audio) <= 6400:
                            raise ValueError('audio_chunk_rejected')
                        self.stamp('audio')
                        await self.emit('audio', epoch, id=sentence_id, chunk=chunk_id,
                                        pcm=base64.b64encode(audio).decode(), rate=16000)
                    else:
                        raise ValueError('synthesis_limit')
                finally:
                    await audio_source.aclose()
                await self.emit('sentence_end', epoch, id=sentence_id)
            else:
                raise ValueError('answer_limit')
        finally:
            await source.aclose()
        await self.emit('reply_done', epoch)

    async def receiver(self):
        from anchor_ai.audio import realtime_frame
        while True:
            message = await asyncio.wait_for(self.ws.receive(), 5)
            if message['type'] == 'websocket.disconnect':
                return
            if message.get('bytes') is not None:
                number, pcm, speech = realtime_frame(message['bytes'])
                if number != self.input_sequence:
                    self.stamp('frame_gap', expected=self.input_sequence, received=number)
                    raise ValueError('frame_gap')
                self.input_sequence += 1
                if speech and not self.pcm:
                    await self.interrupt()
                    self.stamp('speech_start')
                if not speech and not self.pcm:
                    continue
                self.pcm.extend(pcm)
                self.silence = 0 if speech else self.silence + 1
                if len(self.pcm) > 960000:
                    raise ValueError('utterance_limit')
                if self.silence == 10:
                    self.turns += 1
                    if self.turns > 2:
                        raise ValueError('session_turn_budget')
                    self.stamp('end_of_speech')
                    utterance = bytes(self.pcm[:-6400])
                    self.pcm.clear()
                    self.silence = 0
                    self.reply = asyncio.create_task(self.response(
                        utterance, [dict(row) for row in self.evidence], self.epoch))
            else:
                text = message.get('text', '')
                if len(text) > 2048:
                    raise ValueError('message_limit')
                item = json.loads(text)
                if item == {'type': 'ping'}:
                    await self.direct(dict(type='pong', epoch=self.epoch))
                elif set(item) == {'type', 'attachment_id'} and item['type'] == 'attachment':
                    if type(item['attachment_id']) is not int:
                        raise ValueError('attachment_id_rejected')
                    rows = self.loader(self.scope, item['attachment_id'])
                    if not isinstance(rows, list) or len(rows) > 16:
                        raise ValueError('evidence_limit')
                    for row in rows:
                        if (row['tenant_id'] != self.scope['tenant_id']
                                or row['ticket_id'] != self.scope['ticket_id']
                                or row.get('untrusted') is not True):
                            raise PermissionError('attachment_not_owned')
                    self.evidence = rows
                    await self.direct(dict(type='context_ready' if rows else 'context_pending',
                                           epoch=self.epoch))
                elif set(item) == {'type', 'epoch', 'id'} and item['type'] == 'played':
                    async with self.lock:
                        if (item['epoch'] == self.epoch and type(item['id']) is int
                                and item['id'] == self.ack + 1 and item['id'] in self.completed):
                            self.ack = item['id']
                else:
                    raise ValueError('event_rejected')

    async def supervisor(self):
        while True:
            await asyncio.sleep(0.005)
            if self.kill_file.exists():
                self.reason = 'human_only'
                return
            if time.monotonic() - self.started >= 30:
                self.reason = 'session_deadline'
                return
            if self.reply and self.reply.done():
                self.reply.result()

    async def run(self):
        from anchor_ai.streaming import cancel_tasks
        tasks = [asyncio.create_task(f()) for f in
                 (self.receiver, self.writer, self.supervisor)]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        except Exception:
            self.reason = 'policy_or_stage_failure'
            self.stamp('session_failure')
        finally:
            await cancel_tasks([*tasks, self.reply])
            payload = self.handoff(dict(scope=self.scope, reason=self.reason,
                transcript=self.messages, heard=self.heard + self.sentences[:self.ack],
                evidence=[{k: row[k] for k in ('attachment_id', 'page', 'region')}
                          for row in self.evidence], pending_action='none_executed',
                approval_required=True))
            self.ledger.finish(self.session, self.scope, payload)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self.ws.close(code=1000), 0.1)
            self.stamp('session_closed')


def register(application, *, ledger, ports, loader, handoff, kill_file):
    traces = []
    application.state.realtime_traces = traces

    @application.websocket('/v1/realtime')
    async def realtime(ws: WebSocket):
        await ws.accept()
        session = uuid.uuid4().hex
        try:
            if Path(kill_file).exists():
                await ws.close(code=1013)
                return
            text = await asyncio.wait_for(ws.receive_text(), 1)
            if len(text) > 2048:
                raise PermissionError('credential_rejected')
            first = json.loads(text)
            if set(first) != {'type', 'credential'} or first['type'] != 'authenticate':
                raise PermissionError('credential_rejected')
            scope = ledger.redeem(first['credential'], ws.headers.get('origin'), session)
        except OverflowError:
            await ws.close(code=1013)
            return
        except Exception:
            await ws.close(code=1008)
            return
        trace = []
        traces.append(trace)
        interaction = Interaction(ws, scope, session, ledger, ports, loader, kill_file, trace, handoff)
        try:
            await interaction.direct(dict(type='authenticated', epoch=0))
        except Exception:
            ledger.finish(session, scope, handoff(dict(scope=scope, reason='authentication_delivery_failed',
                transcript=[], heard=[], evidence=[], pending_action='none_executed', approval_required=True)))
            with contextlib.suppress(Exception):
                await ws.close(code=1013)
            return
        await interaction.run()
    return application


def fixture_app(path, kill_file):
    from anchor_ai.escalation import realtime_handoff
    return register(FastAPI(), ledger=Ledger(path), ports=FixturePorts(), loader=fixture_loader,
                    handoff=realtime_handoff, kill_file=kill_file)


def configure_product(application):
    from anchor_ai.escalation import realtime_handoff
    import importlib
    adapter = os.environ.get('ANCHOR_REALTIME_ADAPTER')
    if not adapter:
        @application.websocket('/v1/realtime')
        async def unavailable(ws: WebSocket):
            await ws.close(code=1013)
        return application
    integration = importlib.import_module(adapter).build()
    if getattr(integration['ports'], 'offline', True):
        raise RuntimeError('live_realtime_adapter_required')
    return register(application, ledger=integration['ledger'], ports=integration['ports'],
                    loader=integration['loader'], handoff=realtime_handoff,
                    kill_file=os.environ['ANCHOR_REALTIME_KILL_FILE'])


def product_app():
    from anchor_ai.app import app
    return app
