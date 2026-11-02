import asyncio
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import pytest

from anchor_ai.checkpoints import CheckpointStore, signed_token
from anchor_ai.contracts import ToolScope
from anchor_ai.tools import DurableOfflineOrderBackend

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / 'labs/langgraph-workflows/run_graph.py'
SECRET = 'local-graph-tests-secret-at-least-32-bytes'


def token(tenant=1, actor='local-reviewer', role='refund_reviewer'):
    return signed_token({'tenant_id': tenant, 'actor_id': actor, 'role': role,
                         'exp': int(time.time()) + 600}, SECRET)


def run_cli(db, *args, credential=None, success=True):
    # Other tests may select providers, policies, or credentials in the parent.
    env = {key: value for key, value in os.environ.items()
           if not key.startswith('ANCHOR_')}
    env.update(ANCHOR_LLM_PROVIDER='fake', ANCHOR_APPROVAL_SECRET=SECRET,
               ANCHOR_GRAPH_TOKEN=credential or token())
    result = subprocess.run([sys.executable, str(CLI), '--db', str(db), *args],
                            cwd=ROOT, env=env, text=True, capture_output=True, timeout=8)
    if success:
        assert result.returncode == 0, result.stderr + result.stdout
    else:
        assert result.returncode != 0
    return result.stdout


def copy_migrated_database(destination):
    # Minimal schema fixture matching the Rails migration. Controller behavior
    # and migration integration are exercised by the Rails test suite.
    with sqlite3.connect(destination) as db:
        db.executescript('''
            CREATE TABLE agent_checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id INTEGER NOT NULL, conversation_id INTEGER NOT NULL,
                thread_id TEXT NOT NULL UNIQUE, action_key TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending', context_json TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE INDEX checkpoint_scope ON agent_checkpoints(tenant_id, conversation_id);
            CREATE TABLE agent_approvals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_checkpoint_id INTEGER NOT NULL UNIQUE
                    REFERENCES agent_checkpoints(id),
                tenant_id INTEGER NOT NULL, actor_id TEXT NOT NULL,
                action_key TEXT NOT NULL, nonce TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL DEFAULT 'approved',
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        ''')


def test_fresh_process_resume_and_idempotency(tmp_path, monkeypatch):
    monkeypatch.setenv('ANCHOR_APPROVAL_SECRET', SECRET)
    db = tmp_path / 'workflow.sqlite3'
    copy_migrated_database(db)
    first = run_cli(db, '--ticket', 'labs/langgraph-workflows/refund_over_limit.json')
    assert first.count('ENTER triage') == 1
    thread = next(line.split('=', 1)[1] for line in first.splitlines()
                  if line.startswith('thread_id='))
    store = CheckpointStore(db)
    from anchor_ai.graphs import resume_refund
    with pytest.raises(PermissionError, match='authorized_approval_required'):
        resume_refund(db, thread, token())
    with pytest.raises(PermissionError, match='review_denied'):
        store.inspect(thread, token(tenant=2))
    with pytest.raises(PermissionError, match='review_denied'):
        store.approval(thread, token(role='customer'))
    # Seed only the authenticated controller's durable output. Its request,
    # authorization, action binding, and replay tests live in the Rails suite.
    with sqlite3.connect(db) as connection:
        connection.execute('''INSERT INTO agent_approvals
            (agent_checkpoint_id,tenant_id,actor_id,action_key,nonce,status,created_at,updated_at)
            SELECT id,tenant_id,'local-reviewer',action_key,'test-once','approved',
                   CURRENT_TIMESTAMP,CURRENT_TIMESTAMP
            FROM agent_checkpoints WHERE thread_id=?''', (thread,))
        connection.execute("UPDATE agent_checkpoints SET status='approved' WHERE thread_id=?", (thread,))
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute('''INSERT INTO agent_approvals
                SELECT NULL,agent_checkpoint_id,tenant_id,actor_id,action_key,
                       'replayed-nonce',status,created_at,updated_at FROM agent_approvals''')
    with pytest.raises(PermissionError, match='authorized_approval_required'):
        resume_refund(db, thread, token(actor='another-reviewer'))
    resumed = run_cli(db, '--resume', thread)
    assert 'ENTER triage' not in resumed
    assert 'ENTER specialist' in resumed
    assert 'triage called once' in resumed
    assert '["triage", "approved", "specialist", "escalation"]' in resumed
    with pytest.raises(PermissionError, match='approval_replay_rejected'):
        resume_refund(db, thread, token())
    with sqlite3.connect(db) as connection:
        row = connection.execute('SELECT status,context_json,action_key FROM agent_checkpoints WHERE thread_id=?',
                                 (thread,)).fetchone()
        assert row[0] == 'completed'
        assert 'alex@example.test' not in row[1]
        assert '[REDACTED_EMAIL]' in row[1]
        assert connection.execute('SELECT count(*) FROM graph_tool_results').fetchone()[0] == 1
        assert connection.execute('SELECT status FROM agent_approvals WHERE action_key=?',
                                  (row[2],)).fetchone()[0] == 'consumed'
    scope = ToolScope(tenant_id=1, conversation_id=7)
    backend = DurableOfflineOrderBackend(db)
    kwargs = dict(scope=scope, order_id='A100', amount_cents=500, idempotency_key=row[2])
    assert asyncio.run(backend.issue_refund(**kwargs))['status'] == 'refunded'
    assert backend.invocations == []
    with pytest.raises(ValueError, match='idempotency_conflict'):
        asyncio.run(backend.issue_refund(**{**kwargs, 'amount_cents': 501}))
    with sqlite3.connect(db) as connection:
        assert connection.execute('SELECT count(*) FROM graph_tool_results').fetchone()[0] == 1
    print('Verified: fresh-process continuation, authorized approval, one triage, one durable refund action.')
