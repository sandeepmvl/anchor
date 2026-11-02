import asyncio
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from anchor_ai.tests.test_escalation import scenario, submit
from anchor_ai.escalation import EscalationService


async def run(path):
    service = EscalationService(path)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT name FROM sqlite_master WHERE name='escalations'").fetchone()
    print('STORAGE: production adapter created escalations in temporary SQLite')
    rejected_parts = scenario(path, conversation_id=1)
    service, backend, registry, call, scope, state, reviewer = rejected_parts
    checkpoint = await submit(rejected_parts)
    record = service.inspect(checkpoint, reviewer)
    assert backend.count() == 0
    assert 'alice@example.com' not in json.dumps(record)
    print('PAUSED:', json.dumps(record, sort_keys=True))
    stranger = reviewer.model_copy(update={'tenant_id': reviewer.tenant_id + 1000})
    try:
        await service.resolve(checkpoint, stranger, 'approve', registry=registry, execute=backend.execute)
        raise AssertionError('cross-tenant review was accepted')
    except PermissionError:
        assert service.inspect(checkpoint, reviewer)['status'] == 'pending'
        print('DENIED: different tenant; checkpoint remains pending')
    rejected = await service.resolve(checkpoint, reviewer, 'reject', reason='Proof of purchase is missing.')
    assert backend.count() == 0 and backend.invocations == []
    assert rejected['outcome']['human_handoff']
    print('REJECTED:', rejected['outcome']['reply'], 'refunds=0')
    approved_parts = scenario(path, conversation_id=2)
    service, backend, registry, call, scope, state, reviewer = approved_parts
    checkpoint = await submit(approved_parts)
    key = service.inspect(checkpoint, reviewer)['action_key']
    service = EscalationService(path)
    approved = await service.resolve(checkpoint, reviewer, 'approve', registry=registry, execute=backend.execute)
    repeated = await service.resolve(checkpoint, reviewer, 'approve', registry=registry, execute=backend.execute)
    assert approved['result']['ok']
    assert repeated['deduplicated']
    assert backend.count() == 1 and backend.invocations == [key]
    print('APPROVED:', approved['outcome']['reply'], 'refunds=1')
    print('DUPLICATE: existing result returned; resumed actions=1; original action key preserved')
    print('PASS: same over-limit ticket hands off evidence, waits, and follows only the review decision')


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as directory:
        asyncio.run(run(Path(directory) / 'escalation.sqlite3'))
