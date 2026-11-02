import base64
import hashlib
import hmac
import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager

from .pii import redact_graph


def signed_token(claims, secret):
    encoded = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode()
    signature = hmac.new(secret.encode(), encoded.encode(), hashlib.sha256).hexdigest()
    return encoded + '.' + signature


def actor_claims(token):
    try:
        secret = os.environ['ANCHOR_APPROVAL_SECRET']
        if len(secret) < 32:
            raise ValueError()
        encoded, signature = token.split('.', 1)
        expected = hmac.new(secret.encode(), encoded.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise ValueError()
        claims = json.loads(base64.urlsafe_b64decode(encoded))
        if (type(claims['tenant_id']) is not int or claims['tenant_id'] <= 0 or
                claims['role'] != 'refund_reviewer' or
                not isinstance(claims['actor_id'], str) or not claims['actor_id'] or
                type(claims['exp']) is not int or claims['exp'] <= time.time()):
            raise ValueError()
        return claims
    except Exception:
        raise PermissionError('review_denied') from None


def environment_token():
    token = os.getenv('ANCHOR_GRAPH_TOKEN')
    if token:
        actor_claims(token)
        return token
    if os.getenv('ANCHOR_GRAPH_DEMO') != '1':
        raise PermissionError('signed_actor_token_required')
    return signed_token({'tenant_id': 1, 'actor_id': 'local-reviewer',
                         'role': 'refund_reviewer', 'exp': int(time.time()) + 600},
                        os.environ['ANCHOR_APPROVAL_SECRET'])


def action_key(tenant_id, conversation_id, order_id, amount_cents):
    material = [tenant_id, conversation_id, 'issue_refund', order_id, amount_cents]
    return hashlib.sha256(json.dumps(material).encode()).hexdigest()


class CheckpointStore:
    def __init__(self, path):
        self.path = str(path)

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def available(self):
        with self.db() as db:
            db.execute('SELECT id FROM agent_checkpoints LIMIT 1')
            db.execute('SELECT id FROM agent_approvals LIMIT 1')
        return 'Checkpoint storage available; offline graph service ready.'

    def pending(self, thread, ticket, key, context):
        safe = json.dumps(redact_graph(context), sort_keys=True)
        with self.db() as db:
            db.execute('''INSERT INTO agent_checkpoints
                (tenant_id,conversation_id,thread_id,action_key,status,context_json,created_at,updated_at)
                VALUES (?,?,?,?,'pending',?,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)
                ON CONFLICT(thread_id) DO NOTHING''',
                (ticket['tenant_id'], ticket['conversation_id'], thread, key, safe))
            row = db.execute('SELECT * FROM agent_checkpoints WHERE thread_id=?', (thread,)).fetchone()
            if (row['tenant_id'] != ticket['tenant_id'] or row['action_key'] != key or
                    row['context_json'] != safe):
                raise ValueError('checkpoint_conflict')

    def inspect(self, thread, token):
        claims = actor_claims(token)
        with self.db() as db:
            row = db.execute('SELECT * FROM agent_checkpoints WHERE thread_id=? AND tenant_id=?',
                             (thread, claims['tenant_id'])).fetchone()
        if row is None:
            raise PermissionError('review_denied')
        return dict(row)

    def approval(self, thread, token, expected_key=None, reference=None):
        row = self.inspect(thread, token)
        claims = actor_claims(token)
        if expected_key is not None and row['action_key'] != expected_key:
            raise PermissionError('action_mismatch')
        with self.db() as db:
            approval = db.execute('''SELECT * FROM agent_approvals
                WHERE agent_checkpoint_id=? AND tenant_id=? AND actor_id=? AND action_key=?''',
                (row['id'], claims['tenant_id'], claims['actor_id'], row['action_key'])).fetchone()
        if (approval is None or row['status'] not in ('approved', 'completed') or
                approval['status'] not in ('approved', 'consumed') or
                (reference is not None and approval['id'] != reference)):
            raise PermissionError('authorized_approval_required')
        return dict(approval)

    def complete(self, thread, token, key):
        approval = self.approval(thread, token, key)
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute("UPDATE agent_checkpoints SET status='completed',updated_at=CURRENT_TIMESTAMP WHERE thread_id=?",
                       (thread,))
            db.execute("UPDATE agent_approvals SET status='consumed',updated_at=CURRENT_TIMESTAMP WHERE id=?",
                       (approval['id'],))

    @contextmanager
    def lease(self, thread):
        # Local execution is bounded to well below this lease. Shared workers
        # need renewable leases and fencing before using a server checkpointer.
        owner = str(uuid.uuid4())
        with self.db() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS graph_leases
                (thread_id TEXT PRIMARY KEY, owner TEXT NOT NULL, expires REAL NOT NULL)''')
            db.execute('BEGIN IMMEDIATE')
            db.execute('DELETE FROM graph_leases WHERE expires < ?', (time.time(),))
            try:
                db.execute('INSERT INTO graph_leases VALUES (?,?,?)',
                           (thread, owner, time.time() + 60))
            except sqlite3.IntegrityError:
                raise PermissionError('thread_busy') from None
        try:
            yield
        finally:
            with self.db() as db:
                db.execute('DELETE FROM graph_leases WHERE thread_id=? AND owner=?', (thread, owner))
