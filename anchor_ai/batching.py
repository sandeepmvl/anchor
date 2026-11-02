"""Bounded bulk boundary. Streaming does not enter this module.

Provider submission is not durable-task completion. An AiTask adapter must
retrieve and commit required results before advancing its progress checkpoint.
Retry-safe provider submission requires downstream idempotency enforcement.
"""
import asyncio
import hashlib
import hmac
import json
import math
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import yaml
from fastapi import Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field


class Overloaded(Exception):
    pass


class BatchItem(BaseModel):
    model_config = ConfigDict(extra='forbid')
    request_id: str = Field(min_length=1, max_length=80)
    tenant_id: int = Field(gt=0)
    conversation_id: int = Field(default=0, ge=0)
    operation: str = Field(pattern='^(embedding|inference)$')
    model: str = Field(min_length=1, max_length=80)
    revision: str = Field(min_length=1, max_length=80)
    settings: dict = Field(default_factory=dict)
    text: str = Field(min_length=1, max_length=4096)

    def key(self):
        settings = json.dumps(self.settings, sort_keys=True, allow_nan=False,
                              separators=(',', ':'))
        if len(settings.encode()) > 1024:
            raise ValueError('settings_too_large')
        # Conversations are partitioned as well as tenants: no shared history.
        return (self.tenant_id, self.conversation_id, self.operation,
                self.model, self.revision, settings)


class BatchEnvelope(BaseModel):
    model_config = ConfigDict(extra='forbid')
    items: list[BatchItem] = Field(min_length=1, max_length=128)
    asynchronous: bool = False


def configuration(path='config/batching.yml'):
    c = yaml.safe_load(Path(path).read_text())['batching']
    limits = {'size': 64, 'window_ms': 1000, 'max_pending': 4096,
              'provider_concurrency': 32, 'unit_cap': 100000,
              'max_input_bytes': 65536, 'call_timeout_ms': 60000}
    for name, maximum in limits.items():
        if type(c.get(name)) is not int or not 1 <= c[name] <= maximum:
            raise ValueError('invalid_batch_configuration')
    if (c['size'] > c['max_pending'] or c.get('record_content') is not False
            or c.get('budget_basis') != 'whitespace_test_units'):
        raise ValueError('invalid_batch_configuration')
    return c


@dataclass
class Pending:
    item: BatchItem
    identity: str
    units: int
    arrived: float
    future: asyncio.Future


class Scheduler:
    def __init__(self, adapter, config, *, count=None, emit=None):
        self.adapter, self.config = adapter, dict(config)
        self.count = count or (lambda text, key: len(text.split()))
        self.emit = emit or (lambda record: None)
        self.queues = {}
        self.running = set()
        self.pending = 0
        self.closed = False
        self.wake = asyncio.Event()
        self.worker = None

    def validate(self, item):
        key = item.key()
        units = self.count(item.text, key)
        if (not item.text.strip() or type(units) is not int or units <= 0
                or units > self.config['unit_cap']
                or len(item.text.encode()) > self.config['max_input_bytes']):
            raise ValueError('invalid_batch_input')
        return key, units

    async def submit(self, item):
        key, units = self.validate(item)
        if self.closed:
            raise Overloaded('scheduler_closed')
        if self.pending >= self.config['max_pending']:
            raise Overloaded('batch_overloaded')
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        p = Pending(item, uuid.uuid4().hex, units, loop.time(), future)
        self.queues.setdefault(key, []).append(p)
        self.pending += 1
        if self.worker is None:
            self.worker = asyncio.create_task(self._pump())
        self.wake.set()
        try:
            return await future
        finally:
            # Cancelling one waiter never cancels the shared provider call.
            self.wake.set()

    async def _pump(self):
        loop = asyncio.get_running_loop()
        while True:
            self.wake.clear()
            for key, queue in list(self.queues.items()):
                live = [p for p in queue if not p.future.cancelled()]
                self.pending -= len(queue) - len(live)
                if live:
                    self.queues[key] = live
                else:
                    del self.queues[key]
            now = loop.time()
            for key, queue in sorted(list(self.queues.items()),
                                     key=lambda pair: pair[1][0].arrived):
                if len(self.running) >= self.config['provider_concurrency']:
                    break
                batch, used = [], 0
                for p in queue:
                    if (len(batch) == self.config['size']
                            or used + p.units > self.config['unit_cap']):
                        break
                    batch.append(p)
                    used += p.units
                deadline = queue[0].arrived + self.config['window_ms'] / 1000
                ready = (len(batch) == self.config['size']
                         or len(batch) < len(queue) or now >= deadline)
                if not ready:
                    continue
                rest = queue[len(batch):]
                if rest:
                    self.queues[key] = rest
                else:
                    del self.queues[key]
                task = asyncio.create_task(self._deliver(key, batch))
                self.running.add(task)
                task.add_done_callback(self._finished)
            delay = None
            if self.queues and len(self.running) < self.config['provider_concurrency']:
                deadline = min(q[0].arrived for q in self.queues.values())
                delay = max(0, deadline + self.config['window_ms'] / 1000 - loop.time())
            try:
                if delay is None:
                    await self.wake.wait()
                else:
                    await asyncio.wait_for(self.wake.wait(), delay)
            except asyncio.TimeoutError:
                pass

    def _finished(self, task):
        self.running.discard(task)
        self.wake.set()

    async def _deliver(self, key, batch):
        # Telemetry is numeric and content-free. Do not inherit a caller's span.
        delays = [(asyncio.get_running_loop().time() - p.arrived) * 1000 for p in batch]
        self.emit({'batch_size': len(batch), 'queue_ms': delays,
                   'outstanding': self.pending})
        try:
            records = [(p.identity, p.item.text) for p in batch]
            results = await asyncio.wait_for(self.adapter.call(key, records),
                                            self.config['call_timeout_ms'] / 1000)
            if (not isinstance(results, dict)
                    or set(results) != {p.identity for p in batch}):
                raise ValueError('invalid_batch_response')
        except asyncio.CancelledError:
            for p in batch:
                if not p.future.done():
                    p.future.cancel()
            raise
        except Exception:
            results = {p.identity: {'error': 'provider_call_failed'} for p in batch}
        else:
            results = {identity: value if isinstance(value, dict)
                       else {'error': 'invalid_provider_item'}
                       for identity, value in results.items()}
        finally:
            self.pending -= len(batch)
        for p in batch:
            if not p.future.done():
                p.future.set_result(results[p.identity])

    async def close(self):
        self.closed = True
        if self.worker is not None:
            self.worker.cancel()
            await asyncio.gather(self.worker, return_exceptions=True)
        tasks = list(self.running)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for queue in self.queues.values():
            for p in queue:
                if not p.future.done():
                    p.future.cancel()
        self.queues.clear()
        self.pending = 0


class OfflineBatchProvider:
    """Synthetic 12ms/call + 0.1ms/item; no model quality claims."""
    fixture = True

    def __init__(self):
        self.calls = 0
        self.groups = []

    async def call(self, key, records):
        self.calls += 1
        self.groups.append((key, len(records)))
        await asyncio.sleep(.012 + .0001 * len(records))
        result = {}
        for identity, text in reversed(records):
            if text == '!partial_failure':
                result[identity] = {'error': 'provider_item_failed'}
            elif key[2] == 'embedding':
                result[identity] = {'embedding': list(hashlib.sha256(text.encode()).digest()[:8])}
            else:
                result[identity] = {'reply': 'Offline reply: ' + text}
        return result

    async def submit_job(self, key, records):
        # Submission returns an identifier, not completed results.
        return 'fake_batch_' + uuid.uuid4().hex


class EmbeddingArrayProvider:
    """Real array adapter over Anchor's existing validated embedding client."""
    fixture = False

    def __init__(self, *, provider, config, transport=None, api_key=None):
        from anchor_ai.embeddings import validate_config
        validate_config(provider, config['providers'][provider])
        if provider == 'offline':
            raise ValueError('live_embedding_provider_required')
        self.provider, self.config = provider, config
        self.transport, self.api_key = transport, api_key

    async def call(self, key, records):
        from anchor_ai.embeddings import embed
        settings = self.config['providers'][self.provider]
        requested = json.loads(key[5])
        if (key[2] != 'embedding' or key[3] != settings['model']
                or key[4] != settings['revision']
                or set(requested) != {'input_type'}):
            raise ValueError('unsupported_bulk_capability')
        result = await asyncio.to_thread(
            embed, [text for _, text in records], requested['input_type'],
            provider=self.provider, config=self.config,
            transport=self.transport, api_key=self.api_key)
        return {identity: {'embedding': vector}
                for (identity, _), vector in zip(records, result.vectors)}


class BatchRuntime:
    """verify(token, items) must authorize EVERY tenant and conversation."""
    def __init__(self, scheduler, *, verify, path, production=False,
                 evidence=None, tokenizer=None):
        required = {'tenant_authorization', 'provider_limits', 'load_test',
                    'interactive_capacity', 'real_tokenizer'}
        if production and (getattr(scheduler.adapter, 'fixture', True)
                           or tokenizer is None or not evidence
                           or any(evidence.get(k) is not True for k in required)):
            raise ValueError('batch_release_evidence_missing')
        if tokenizer is not None:
            scheduler.count = tokenizer
        self.scheduler, self.verify = scheduler, verify
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute('CREATE TABLE IF NOT EXISTS provider_batch_jobs ('
                       'job TEXT PRIMARY KEY, tenant INTEGER, conversation INTEGER, '
                       'provider_job TEXT NOT NULL, created REAL)')

    async def execute(self, token, envelope):
        if not token or self.verify(token, envelope.items) is not True:
            raise PermissionError('batch_scope_rejected')
        # Authorize and validate the entire envelope before dispatching anything.
        for item in envelope.items:
            self.scheduler.validate(item)
        identities = [(p.tenant_id, p.conversation_id, p.request_id) for p in envelope.items]
        if len(set(identities)) != len(identities):
            raise ValueError('duplicate_batch_identity')
        if envelope.asynchronous:
            return await self._jobs(envelope.items)

        async def one(item):
            try:
                outcome = await self.scheduler.submit(item)
            except Overloaded:
                outcome = {'error': 'batch_overloaded', 'retryable': True}
            return {'request_id': item.request_id, 'tenant_id': item.tenant_id,
                    'conversation_id': item.conversation_id, **outcome}
        return {'items': await asyncio.gather(*(one(p) for p in envelope.items))}

    async def _jobs(self, items):
        # A persisted provider identifier is a submission receipt, not a result.
        # Callers must not advance AiTask completion on this return value.
        submit = getattr(self.scheduler.adapter, 'submit_job', None)
        if submit is None:
            raise ValueError('async_batch_capability_unavailable')
        groups = {}
        for item in items:
            groups.setdefault(item.key(), []).append(item)
        jobs = []
        for key, group in groups.items():
            provider_id = await asyncio.wait_for(
                submit(key, [(p.request_id, p.text) for p in group]),
                self.scheduler.config['call_timeout_ms'] / 1000)
            if not isinstance(provider_id, str) or not provider_id:
                raise ValueError('invalid_provider_job_identifier')
            job = uuid.uuid4().hex
            with sqlite3.connect(self.path) as db:
                db.execute('INSERT INTO provider_batch_jobs VALUES(?,?,?,?,?)',
                           (job, key[0], key[1], provider_id, time.time()))
            jobs.append({'job_id': job, 'provider_job_id': provider_id,
                         'tenant_id': key[0], 'conversation_id': key[1]})
        return {'jobs': jobs, 'status': 'submitted_not_completed'}

    def polling_identifier(self, token, item, job):
        if not token or self.verify(token, [item]) is not True:
            raise PermissionError('batch_scope_rejected')
        with sqlite3.connect(self.path) as db:
            row = db.execute('SELECT provider_job FROM provider_batch_jobs '
                             'WHERE job=? AND tenant=? AND conversation=?',
                             (job, item.tenant_id, item.conversation_id)).fetchone()
        if row is None:
            raise PermissionError('job_not_in_scope')
        return row[0]

    async def close(self):
        await self.scheduler.close()


def register_batch_routes(app):
    @app.post('/v1/internal/batch')
    async def internal_batch(body: BatchEnvelope, request: Request,
                             x_anchor_secret: str | None = Header(default=None),
                             x_anchor_context: str | None = Header(default=None)):
        expected = os.environ.get('ANCHOR_AI_SECRET')
        if not expected:
            raise HTTPException(503, 'service_not_configured')
        if not hmac.compare_digest((x_anchor_secret or '').encode(), expected.encode()):
            raise HTTPException(401, 'service_auth_failed')
        runtime = getattr(request.app.state, 'batch_runtime', None)
        if runtime is None:
            raise HTTPException(503, 'batch_not_configured')
        try:
            return await runtime.execute(x_anchor_context, body)
        except PermissionError:
            raise HTTPException(403, 'batch_scope_rejected') from None
        except ValueError:
            raise HTTPException(422, 'batch_request_rejected') from None
