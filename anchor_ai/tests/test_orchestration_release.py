import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from eval.multi_agent_gate import ROOT, check_preflight, load_adapter
from eval.multi_agent_judge import ReleaseRejected, fingerprint, recorded_judgment
from ops.deploy.orchestration_release import ConfigurationConflict, OrchestrationStore
from anchor_ai.orchestration import MilestoneUnavailable, run_milestone


def test_missing_product_composition_fails_closed(monkeypatch):
    monkeypatch.delenv('ANCHOR_ORCHESTRATION_RELEASE_ADAPTER', raising=False)
    with pytest.raises(ReleaseRejected, match='product_composition_adapter_missing'):
        load_adapter()
    with pytest.raises(MilestoneUnavailable):
        asyncio.run(run_milestone(None, 'signed-context', {}))


def test_missing_sandbox_isolation_fails_preflight():
    class Unisolated:
        def preflight(self):
            return dict(queue=True, fake_model=True, seeded_retrieval=True,
                        framework=True, local_mcp=True, sandbox_image=True,
                        sandbox_isolation=False)
    with pytest.raises(ReleaseRejected, match='sandbox_isolation'):
        check_preflight(Unisolated())


def test_recorded_judge_regression_blocks_release(tmp_path):
    case = {'id': 'regression-control'}
    evidence = {'answer': 'An unsupported claim.'}
    recording = tmp_path / 'judge.jsonl'
    recording.write_text(json.dumps({
        'case_id': case['id'], 'input_sha256': fingerprint(evidence),
        'judge_model': 'recorded-judge', 'rubric_version': 'multi-agent-v1',
        'grounded': False, 'handoff_complete': True, 'score': 0.2
    }) + '\n')
    with pytest.raises(ReleaseRejected, match='judge_groundedness_regression'):
        recorded_judgment(case, evidence, recording)


def test_judge_recording_cannot_be_reused_for_changed_evidence(tmp_path):
    case = {'id': 'binding-control'}
    recording = tmp_path / 'judge.jsonl'
    recording.write_text(json.dumps({
        'case_id': case['id'], 'input_sha256': fingerprint({'answer': 'old'}),
        'judge_model': 'recorded-judge', 'rubric_version': 'multi-agent-v1',
        'grounded': True, 'handoff_complete': True, 'score': 1
    }) + '\n')
    with pytest.raises(ReleaseRejected, match='judge_replay_miss'):
        recorded_judgment(case, {'answer': 'changed'}, recording)


def test_release_gate_rejects_unavailable_mcp_without_editing_configuration():
    class McpUnavailable:
        def preflight(self):
            return dict(queue=True, fake_model=True, seeded_retrieval=True,
                        framework=True, local_mcp=False, sandbox_image=True,
                        sandbox_isolation=True)
    config = ROOT / 'config/orchestration.yml'
    before = config.read_bytes()
    with pytest.raises(ReleaseRejected, match='preflight_failed_local_mcp'):
        check_preflight(McpUnavailable())
    assert config.read_bytes() == before


def initialized_store(path):
    store = OrchestrationStore(path)
    store.register('prior', {'worker': 'refund', 'max_steps': 6})
    store.register('candidate', {'worker': 'refund', 'max_steps': 5})
    return store


def test_atomic_configuration_rollback_requires_no_app_deployment(tmp_path):
    path = tmp_path / 'shared.sqlite3'
    store = initialized_store(path)
    store.activate('prior', 'candidate', 'release-test')
    stale_worker = OrchestrationStore(path)
    assert stale_worker.snapshot()['version'] == 'candidate'
    store.activate('candidate', 'prior', 'incident-test')
    assert stale_worker.snapshot()['version'] == 'prior'
    assert stale_worker.snapshot()['human_only'] is True
    with pytest.raises(ConfigurationConflict, match='active_version_changed'):
        stale_worker.activate('candidate', 'candidate', 'stale-operator')
    assert store.snapshot()['version'] == 'prior'
    with pytest.raises(ConfigurationConflict, match='immutable_version'):
        store.register('prior', {'worker': 'other'})


def test_human_only_switch_preserves_full_context(tmp_path):
    path = tmp_path / 'shared.sqlite3'
    store = initialized_store(path)
    store.human_only(False, 'test')
    another_worker = OrchestrationStore(path)
    assert another_worker.snapshot()['human_only'] is False
    store.human_only(True, 'incident')
    assert another_worker.snapshot()['human_only'] is True
    context = dict(ticket='Contact customer@example.com', evidence=[{'chunk_id': 901}],
                   completed_actions=['retrieve'], pending_actions=['lookup_order', 'calculate'],
                   approval_state='not_approved', trace_id='a' * 32,
                   orchestration_version='prior', reason='operator_stop')
    saved = another_worker.save_handoff(1, 'R-17', context)
    restored = OrchestrationStore(path).handoff(1, 'R-17')
    assert restored == saved
    assert set(restored) == set(context)
    assert 'customer@example.com' not in json.dumps(restored)
    assert restored['evidence'] == context['evidence']
    assert restored['pending_actions'] == context['pending_actions']
    assert OrchestrationStore(path).handoff(2, 'R-17') is None
    assert context['ticket'] == 'Contact customer@example.com'


def test_smoke_does_not_make_production_gate_pass(monkeypatch):
    env = dict(os.environ)
    env.pop('ANCHOR_ORCHESTRATION_RELEASE_ADAPTER', None)
    result = subprocess.run([sys.executable, 'eval/multi_agent_gate.py'],
                            cwd=ROOT, env=env, capture_output=True, text=True, timeout=5)
    assert result.returncode != 0
    assert 'product_composition_adapter_missing' in result.stdout
