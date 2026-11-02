import json
import time
from collections import OrderedDict
from pathlib import Path

import yaml

from anchor_ai.context import count, clip
from anchor_ai.pii import redact_memory
from anchor_ai.storage import connect_memory

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    return yaml.safe_load((ROOT / 'config/agent_memory.yml').read_text())


def scope_ids(tenant_id, conversation_id):
    if any(type(value) is not int or value <= 0
           for value in (tenant_id, conversation_id)):
        raise ValueError('invalid_memory_scope')
    return tenant_id, conversation_id


class MemoryStore:
    def __init__(self, path, *, clock=time.time, config=None):
        self.config = config or configuration()
        self.clock = clock
        self.working = OrderedDict()
        for section, names in (
            ('working', ('max_entries', 'max_scopes', 'entry_tokens')),
            ('facts', ('ttl_seconds',)),
            ('episodes', ('ttl_seconds', 'max_entries', 'summary_tokens')),
            ('recall', ('token_budget',)),
        ):
            if any(type(self.config[section][name]) is not int or
                   self.config[section][name] <= 0 for name in names):
                raise ValueError('invalid_memory_config')
        keys = self.config['facts']['keys']
        if not keys or len(keys) != len(set(keys)) or any(
                not isinstance(key, str) or not key.isidentifier() for key in keys):
            raise ValueError('invalid_memory_keys')
        self.db = connect_memory(path)

    def add_working(self, tenant_id, conversation_id, text):
        scope = scope_ids(tenant_id, conversation_id)
        safe = clip(redact_memory(text), self.config['working']['entry_tokens'])
        entries = self.working.pop(scope, []) + [safe]
        self.working[scope] = entries[-self.config['working']['max_entries']:]
        while len(self.working) > self.config['working']['max_scopes']:
            self.working.popitem(last=False)
        return list(self.working[scope])

    def working_context(self, tenant_id, conversation_id):
        scope = scope_ids(tenant_id, conversation_id)
        return list(self.working.get(scope, []))

    def _prune(self, tenant_id, conversation_id, now):
        for table in ('agent_memories', 'episodic_memory_log'):
            self.db.execute(
                f'DELETE FROM {table} WHERE tenant_id=? AND conversation_id=? '
                'AND expires_at<=?', (tenant_id, conversation_id, now))
        self.db.execute('''DELETE FROM episodic_memory_log
            WHERE tenant_id=? AND conversation_id=? AND id NOT IN (
                SELECT id FROM episodic_memory_log
                WHERE tenant_id=? AND conversation_id=?
                ORDER BY created_at DESC, id DESC LIMIT ?)''',
            (tenant_id, conversation_id, tenant_id, conversation_id,
             self.config['episodes']['max_entries']))

    def maintain(self, tenant_id, conversation_id):
        scope_ids(tenant_id, conversation_id)
        with self.db:
            self._prune(tenant_id, conversation_id, self.clock())

    def _episode(self, tenant_id, conversation_id, summary, now):
        safe = clip(redact_memory(summary), self.config['episodes']['summary_tokens'])
        self.db.execute('''INSERT INTO episodic_memory_log
            (tenant_id,conversation_id,summary,created_at,expires_at)
            VALUES(?,?,?,?,?)''', (tenant_id, conversation_id, safe, now,
            now + self.config['episodes']['ttl_seconds']))

    def log_episode(self, tenant_id, conversation_id, summary):
        scope_ids(tenant_id, conversation_id)
        now = self.clock()
        with self.db:
            self._episode(tenant_id, conversation_id, summary, now)
            self._prune(tenant_id, conversation_id, now)

    def save_memory(self, tenant_id, conversation_id, key, value):
        scope_ids(tenant_id, conversation_id)
        if not isinstance(key, str) or key not in self.config['facts']['keys']:
            raise ValueError('unsupported_memory_key')
        safe = redact_memory(value)
        if key == 'contact_preference' and safe not in ('phone', 'email'):
            raise ValueError('unsupported_contact_preference')
        if count(safe) > self.config['working']['entry_tokens']:
            raise ValueError('memory_value_budget_exceeded')
        now = self.clock()
        with self.db:
            self.db.execute('''INSERT INTO agent_memories
                (tenant_id,conversation_id,key,value,updated_at,expires_at)
                VALUES(?,?,?,?,?,?)
                ON CONFLICT(tenant_id,conversation_id,key) DO UPDATE SET
                value=excluded.value,updated_at=excluded.updated_at,
                expires_at=excluded.expires_at''',
                (tenant_id, conversation_id, key, safe, now,
                 now + self.config['facts']['ttl_seconds']))
            self._episode(tenant_id, conversation_id,
                          'save_memory ' + key + ': ' + safe, now)
            self._prune(tenant_id, conversation_id, now)
        return {'saved': key, 'value': safe}

    def recall(self, tenant_id, conversation_id, *, keys=None, budget=None,
               count_tokens=count):
        scope_ids(tenant_id, conversation_id)
        allowed = self.config['facts']['keys']
        keys = list(allowed if keys is None else keys)
        if any(not isinstance(key, str) or key not in allowed for key in keys):
            raise ValueError('unsupported_memory_key')
        configured = self.config['recall']['token_budget']
        budget = configured if budget is None else budget
        if type(budget) is not int or budget < 0:
            raise ValueError('invalid_memory_budget')
        budget = min(budget, configured)
        self.maintain(tenant_id, conversation_id)
        if not keys:
            return {'memory_block': '', 'omitted': 0}
        placeholders = ','.join('?' for _ in keys)
        rows = self.db.execute(f'''SELECT key,value FROM agent_memories
            WHERE tenant_id=? AND conversation_id=? AND expires_at>?
            AND key IN ({placeholders}) ORDER BY updated_at DESC,key ASC''',
            (tenant_id, conversation_id, self.clock(), *keys)).fetchall()
        retained, block = [], ''
        for row in rows:
            candidate_rows = retained + [dict(row)]
            candidate = json.dumps({'label': 'UNTRUSTED_MEMORY',
                                    'facts': candidate_rows}, ensure_ascii=False)
            if count_tokens(candidate) <= budget:
                retained, block = candidate_rows, candidate
        return {'memory_block': block, 'omitted': len(rows) - len(retained)}

    def recent_steps(self, tenant_id, conversation_id, limit=None):
        scope_ids(tenant_id, conversation_id)
        if limit is None:
            limit = min(5, self.config['episodes']['max_entries'])
        if type(limit) is not int or not 1 <= limit <= self.config['episodes']['max_entries']:
            raise ValueError('invalid_episode_limit')
        self.maintain(tenant_id, conversation_id)
        rows = self.db.execute('''SELECT id,summary,created_at FROM episodic_memory_log
            WHERE tenant_id=? AND conversation_id=? AND expires_at>?
            ORDER BY created_at DESC,id DESC LIMIT ?''',
            (tenant_id, conversation_id, self.clock(), limit)).fetchall()
        return [dict(row) for row in reversed(rows)]

    def forget_conversation(self, tenant_id, conversation_id):
        scope = scope_ids(tenant_id, conversation_id)
        with self.db:
            for table in ('agent_memories', 'episodic_memory_log'):
                self.db.execute(f'DELETE FROM {table} '
                                'WHERE tenant_id=? AND conversation_id=?', scope)
        self.working.pop(scope, None)

    def close(self):
        self.working.clear()
        self.db.close()
