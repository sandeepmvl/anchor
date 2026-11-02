import copy
import json
import sqlite3
from types import SimpleNamespace

import pytest

from anchor_ai.context import agent_memory_prompt, count
from anchor_ai.memory import MemoryStore, configuration


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class FollowUpController:
    """Only a request-boundary double; production code owns recall and packing."""
    def __init__(self, path, clock):
        self.memory = MemoryStore(path, clock=clock)

    def prompt(self, tenant_id, conversation_id):
        scope = SimpleNamespace(tenant_id=tenant_id, conversation_id=conversation_id)
        return agent_memory_prompt('T42 follow-up: what did we discuss about my refund?',
                                   self.memory, scope)


def test_configuration_and_schema(tmp_path):
    config = configuration()
    assert config['working']['max_entries'] == 3
    assert config['facts']['keys'] == ['contact_preference', 'refund_discussion']
    assert config['episodes']['ttl_seconds'] == 86400
    store = MemoryStore(tmp_path / 'standalone.sqlite3')
    tables = {row['name'] for row in store.db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {'agent_memories', 'episodic_memory_log'} <= tables
    assert 'documents' not in tables
    store.close()
    print('Production memory schema created in standalone SQLite; no document tables required.')


def test_upsert_restart_and_isolation(tmp_path):
    path, clock = tmp_path / 'memory.sqlite3', Clock()
    first = FollowUpController(path, clock)
    sql = []
    first.memory.db.set_trace_callback(sql.append)
    first.memory.add_working(1, 42, 'Refund discussion in this controller')
    first.memory.save_memory(1, 42, 'refund_discussion', 'Refund requested; review pending.')
    clock.now += 1
    first.memory.save_memory(1, 42, 'refund_discussion', 'Refund discussed; receipt requested.')
    rows = first.memory.db.execute('''SELECT * FROM agent_memories
        WHERE tenant_id=1 AND conversation_id=42 AND key='refund_discussion' ''').fetchall()
    assert len(rows) == 1
    assert rows[0]['value'] == 'Refund discussed; receipt requested.'
    assert rows[0]['updated_at'] == clock.now
    assert rows[0]['expires_at'] == clock.now + configuration()['facts']['ttl_seconds']
    assert any('ON CONFLICT(TENANT_ID,CONVERSATION_ID,KEY) DO UPDATE SET' in
               ' '.join(statement.upper().split()) for statement in sql)
    first.memory.db.set_trace_callback(None)
    with pytest.raises(sqlite3.IntegrityError):
        with first.memory.db:
            first.memory.db.execute('''INSERT INTO agent_memories
                (tenant_id,conversation_id,key,value,updated_at,expires_at)
                VALUES(1,42,'refund_discussion','duplicate',1,999999)''')
    assert len(first.memory.recent_steps(1, 42)) == 2
    first.memory.close()

    restarted = FollowUpController(path, clock)
    assert restarted.memory.working_context(1, 42) == []
    prompt = restarted.prompt(1, 42)
    assert 'receipt requested' in prompt
    assert 'review pending' not in prompt
    assert 'UNTRUSTED_MEMORY' in prompt
    assert 'receipt requested' not in restarted.prompt(2, 42)
    assert 'receipt requested' not in restarted.prompt(1, 43)
    assert restarted.memory.recent_steps(2, 42) == []
    assert restarted.memory.recent_steps(1, 43) == []
    restarted.memory.save_memory(2, 42, 'refund_discussion', 'Other tenant discussion.')
    assert 'Other tenant' not in restarted.prompt(1, 42)
    assert 'receipt requested' not in restarted.prompt(2, 42)
    restarted.memory.close()
    print('One updated scoped fact, not duplicate facts.')
    print('Anchor recalls the scoped refund discussion after a controller restart; another tenant cannot retrieve it.')


def test_recall_selection_recency_and_budget(tmp_path):
    clock = Clock()
    store = MemoryStore(tmp_path / 'recall.sqlite3', clock=clock)
    store.save_memory(1, 42, 'contact_preference', 'email')
    clock.now += 1
    store.save_memory(1, 42, 'refund_discussion', 'Receipt requested.')
    result = store.recall(1, 42)
    facts = json.loads(result['memory_block'])['facts']
    assert [fact['key'] for fact in facts] == ['refund_discussion', 'contact_preference']
    assert count(result['memory_block']) <= configuration()['recall']['token_budget']
    selected = store.recall(1, 42, keys=['contact_preference'])
    assert json.loads(selected['memory_block'])['facts'] == [
        {'key': 'contact_preference', 'value': 'email'}]
    empty = store.recall(1, 42, budget=0)
    assert empty == {'memory_block': '', 'omitted': 2}
    assert store.recall(1, 42, keys=[]) == {'memory_block': '', 'omitted': 0}
    # Supply a tokenizer-compatible counter; retain whole entries, not fragments.
    constrained = store.recall(1, 42, budget=1, count_tokens=lambda text:
                              len(json.loads(text)['facts']))
    assert len(json.loads(constrained['memory_block'])['facts']) == 1
    assert constrained['omitted'] == 1
    with pytest.raises(ValueError, match='unsupported_memory_key'):
        store.recall(1, 42, keys=['unknown'])
    store.close()


def test_working_eviction_expiration_and_redaction(tmp_path):
    clock, config = Clock(), copy.deepcopy(configuration())
    config['working']['max_scopes'] = 2
    config['episodes']['max_entries'] = 2
    store = MemoryStore(tmp_path / 'retention.sqlite3', clock=clock, config=config)
    for index in range(5):
        store.add_working(1, 42, 'turn-' + str(index))
    assert store.working_context(1, 42) == ['turn-2', 'turn-3', 'turn-4']
    store.add_working(1, 43, 'another scope')
    store.add_working(2, 42, 'third scope')
    assert store.working_context(1, 42) == []
    assert len(store.working) == 2
    store.add_working(1, 42, 'é' * 2000)
    assert all(count(text) <= config['working']['entry_tokens']
               for text in store.working_context(1, 42))
    sensitive = 'Receipt from alex@example.com; call 415-555-0199.'
    store.save_memory(1, 42, 'refund_discussion', sensitive)
    for index in range(4):
        clock.now += 1
        store.log_episode(1, 42, sensitive + ' outcome-' + str(index))
    episodes = store.recent_steps(1, 42, limit=2)
    assert len(episodes) == 2
    assert 'outcome-2' in episodes[0]['summary']
    assert 'outcome-3' in episodes[1]['summary']
    persisted = '\n'.join(row[0] for row in store.db.execute(
        'SELECT value FROM agent_memories UNION ALL SELECT summary FROM episodic_memory_log'))
    assert 'alex@example.com' not in persisted
    assert '415-555-0199' not in persisted
    assert '[REDACTED_EMAIL]' in persisted
    assert '[REDACTED_PHONE]' in persisted
    clock.now += config['episodes']['ttl_seconds']
    assert store.recent_steps(1, 42) == []
    assert store.db.execute('SELECT COUNT(*) FROM episodic_memory_log').fetchone()[0] == 0
    assert store.recall(1, 42)['memory_block']
    clock.now += config['facts']['ttl_seconds']
    assert store.recall(1, 42)['memory_block'] == ''
    assert store.db.execute('SELECT COUNT(*) FROM agent_memories').fetchone()[0] == 0
    store.close()
    print('Working context evicted; expired episodes and facts removed; persisted outcomes contain no raw test PII.')


def test_atomic_write_validation_and_erasure(tmp_path):
    store = MemoryStore(tmp_path / 'atomic.sqlite3', clock=Clock())
    with pytest.raises(ValueError, match='invalid_memory_scope'):
        store.save_memory(True, 42, 'contact_preference', 'email')
    with pytest.raises(ValueError, match='unsupported_memory_key'):
        store.save_memory(1, 42, 'permission', 'admin')
    store.db.execute('''CREATE TRIGGER fail_episode BEFORE INSERT ON episodic_memory_log
        BEGIN SELECT RAISE(ABORT, 'episode_failure'); END''')
    with pytest.raises(sqlite3.IntegrityError):
        store.save_memory(1, 42, 'contact_preference', 'email')
    assert store.db.execute('SELECT COUNT(*) FROM agent_memories').fetchone()[0] == 0
    store.db.execute('DROP TRIGGER fail_episode')
    store.save_memory(1, 42, 'contact_preference', 'email')
    store.save_memory(2, 42, 'contact_preference', 'phone')
    store.add_working(1, 42, 'temporary context')
    store.forget_conversation(1, 42)
    assert store.working_context(1, 42) == []
    assert store.recall(1, 42)['memory_block'] == ''
    assert store.recent_steps(1, 42) == []
    assert 'phone' in store.recall(2, 42)['memory_block']
    store.close()
