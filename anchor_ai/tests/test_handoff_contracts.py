import copy
import importlib
import json
import sqlite3
from unittest.mock import Mock

import pytest

from anchor_ai.contracts import HANDOFF_KEYS

ADAPTERS = ['orchestration', 'graphs', 'group_chat', 'harness', 'native_agent_wrapper']


def fixture():
    return {'tenant_id': 1, 'conversation_id': 17, 'order_id': 'A104',
            'amount_cents': 300, 'provenance': 'verified-order-read:run17:v1',
            'messages': ['Please review.'], 'evidence': ['Verified order lookup.']}


def checkpoint(value, boundary, tmp_path):
    if boundary == 'dispatch':
        return copy.deepcopy(value)
    with sqlite3.connect(tmp_path / 'handoff.sqlite3') as db:
        db.execute('CREATE TABLE IF NOT EXISTS snapshots (value TEXT NOT NULL)')
        db.execute('DELETE FROM snapshots')
        db.execute('INSERT INTO snapshots VALUES (?)', (json.dumps(value),))
        db.commit()
    with sqlite3.connect(tmp_path / 'handoff.sqlite3') as db:
        return json.loads(db.execute('SELECT value FROM snapshots').fetchone()[0])


@pytest.mark.parametrize('adapter', ADAPTERS)
@pytest.mark.parametrize('boundary', ['dispatch', 'resume'])
def test_original_context_survives(adapter, boundary, tmp_path):
    entry = importlib.import_module('anchor_ai.' + adapter).dispatch_specialist
    trusted = fixture()
    received = checkpoint(trusted, boundary, tmp_path)
    invoke = Mock(side_effect=lambda context: {'order_id': context['order_id']})
    assert entry(received, trusted, invoke, boundary=boundary) == {'order_id': 'A104'}
    invoke.assert_called_once_with(trusted)
    # The specialist receives a copy, not authority to mutate the saved snapshot.
    invoke.call_args.args[0]['messages'].append('changed')
    assert trusted == fixture()


@pytest.mark.parametrize('adapter', ADAPTERS)
@pytest.mark.parametrize('boundary', ['dispatch', 'resume'])
@pytest.mark.parametrize('key', HANDOFF_KEYS)
@pytest.mark.parametrize('fault', ['missing', 'changed'])
def test_incomplete_or_stale_state_fails_closed(adapter, boundary, key, fault, tmp_path):
    entry = importlib.import_module('anchor_ai.' + adapter).dispatch_specialist
    trusted = fixture()
    candidate = checkpoint(trusted, boundary, tmp_path)
    if fault == 'missing':
        candidate.pop(key)
    else:
        candidate[key] = {'tenant_id': 2, 'conversation_id': 99, 'order_id': 'B205',
                          'amount_cents': 900, 'provenance': 'model-guessed',
                          'messages': ['Different request'], 'evidence': ['Unverified']}[key]
    invoke = Mock()
    result = entry(candidate, trusted, invoke, boundary=boundary)
    invoke.assert_not_called()
    assert result['status'] == 'human_escalation'
    assert result['tool_calls'] == [] and result['do_not_retry']
    assert key in result['invalid_keys']
    assert result['context']['order_id'] == 'A104'
    assert trusted == fixture()


@pytest.mark.parametrize('adapter', ADAPTERS)
@pytest.mark.parametrize('boundary', ['dispatch', 'resume'])
@pytest.mark.parametrize('bad', ['', None, 104, True])
def test_invalid_order_is_not_reconstructed(adapter, boundary, bad):
    entry = importlib.import_module('anchor_ai.' + adapter).dispatch_specialist
    trusted = fixture()
    candidate = {**trusted, 'order_id': bad, 'model_guess': 'A104'}
    invoke = Mock()
    result = entry(candidate, trusted, invoke, boundary=boundary)
    assert result['status'] == 'human_escalation'
    invoke.assert_not_called()


@pytest.mark.parametrize('adapter', ADAPTERS)
def test_missing_authoritative_state_cannot_be_self_attested(adapter):
    entry = importlib.import_module('anchor_ai.' + adapter).dispatch_specialist
    invoke = Mock()
    result = entry(fixture(), None, invoke, boundary='resume')
    assert result['status'] == 'human_escalation'
    invoke.assert_not_called()
