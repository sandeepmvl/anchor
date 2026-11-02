import asyncio
import copy
import json

import httpx
import pytest
import yaml

from anchor_ai.adapters import MANIFEST, evaluate_release
from ops.deploy.model_release import pin_release, transition


def check_selection(tmp_path):
    state = tmp_path / 'selection.json'
    old = pin_release(state)
    assert old.candidate == 'default-recorded'
    transition('adapter-recorded', path=state)
    candidate = pin_release(state)
    assert candidate.candidate == 'adapter-recorded'
    assert old.complete(old.ticket) != candidate.complete(old.ticket)
    assert old.candidate == 'default-recorded'
    transition(rollback=True, path=state)
    restored = pin_release(state)
    assert restored.candidate == old.candidate
    assert restored.complete(old.ticket) == old.complete(old.ticket)
    assert candidate.complete(old.ticket) != restored.complete(old.ticket)
    before = state.read_bytes()
    with pytest.raises(ValueError, match='unknown_candidate'):
        transition('unapproved', path=state)
    assert state.read_bytes() == before
    manifest = yaml.safe_load(MANIFEST.read_text())
    broken = copy.deepcopy(manifest)
    broken['expected']['adapter_config.json']['lora_alpha'] = 999
    path = tmp_path / 'wrong-hash.yml'
    path.write_text(yaml.safe_dump(broken))
    with pytest.raises(ValueError, match='hash_mismatch'):
        transition('adapter-recorded', path=state, manifest_path=path)
    assert state.read_bytes() == before
    record = copy.deepcopy(manifest['expected_replies'])
    record['recorded_gates']['llm_judge'] = False
    reply_path = tmp_path / 'failed.jsonl'
    reply_path.write_text(json.dumps(record) + '\n')
    broken = copy.deepcopy(manifest)
    broken['replies'] = str(reply_path)
    broken['expected_replies'] = record
    path = tmp_path / 'failed-gate.yml'
    path.write_text(yaml.safe_dump(broken))
    with pytest.raises(ValueError, match='gate_failed'):
        transition('adapter-recorded', path=state, manifest_path=path)
    assert state.read_bytes() == before
    assert evaluate_release('adapter-recorded')['deployment_approved'] is False


def check_provider(tmp_path, monkeypatch):
    from anchor_ai import provider
    monkeypatch.delenv('ANCHOR_MODEL_RELEASE_STATE', raising=False)
    monkeypatch.setenv('ANCHOR_LLM_MODE', 'offline')
    monkeypatch.setenv('ANCHOR_LLM_FIXTURE', '200')
    default = provider.configured_provider()
    assert type(default) is provider.ReplayProvider
    assert default.sequence == provider.ReplayProvider('200').sequence
    state = tmp_path / 'provider-selection.json'
    monkeypatch.setenv('ANCHOR_MODEL_RELEASE_STATE', str(state))
    before = provider.configured_provider()
    transition('adapter-recorded', path=state)
    selected = provider.configured_provider()
    payload = {'messages': [{'role': 'user', 'content': selected.selection.ticket}]}
    response = selected.send(httpx.Request('POST', 'https://offline.invalid', json=payload))
    assert response.json()['model'] == 'adapter-recorded'
    assert response.json()['choices'][0]['message']['content'] == selected.selection.reply
    transition(rollback=True, path=state)
    assert before.selection.candidate == 'default-recorded'
    assert selected.selection.candidate == 'adapter-recorded'
    assert provider.configured_provider().selection.candidate == 'default-recorded'
    async def collect():
        return ''.join([delta.text async for delta in provider.stream_tokens(
            selected, selected.selection.ticket)])
    assert asyncio.run(collect()) == selected.selection.reply
    monkeypatch.delenv('ANCHOR_MODEL_RELEASE_STATE')
    assert type(provider.configured_provider()) is provider.ReplayProvider
