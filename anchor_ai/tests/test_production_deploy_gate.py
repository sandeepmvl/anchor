import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

import runpy

# Load the extensionless deployment command without invoking its CLI.
launch = SimpleNamespace(**runpy.run_path(str(Path(__file__).resolve().parents[2] / 'bin/deploy')))
from ops.deploy.validate_manifest import launch_policy, release_gate, render
from eval import release_gate as evaluation


def test_each_failed_gate_stops_following_commands():
    for failed in range(5):
        seen = []
        commands = [[str(i)] for i in range(5)]

        def runner(command, **kwargs):
            seen.append(command)
            return SimpleNamespace(returncode=int(command == commands[failed]))

        with pytest.raises(ValueError, match='candidate gate failed'):
            launch.run_checks(commands, runner)
        assert seen == commands[:failed + 1]


def test_failed_candidate_preserves_prior_release(tmp_path):
    path = tmp_path / 'rehearsal.json'
    launch.promote_rehearsal(path, 'stable', lambda: [{'passed': True}])
    before = path.read_bytes()

    def failed():
        raise ValueError('injection gate failed')

    with pytest.raises(ValueError):
        launch.promote_rehearsal(path, 'unsafe', failed)
    assert path.read_bytes() == before
    assert json.loads(before)['release'] == 'stable'


def test_rehearsal_receipt_is_not_production_evidence(tmp_path):
    path = tmp_path / 'receipt.json'
    receipt = launch.promote_rehearsal(path, 'fixture', lambda: [])
    assert receipt['deployment_approved'] is False
    with pytest.raises(ValueError):
        evaluation.require_launch_evidence({}, evidence=path)


def test_missing_infrastructure_cannot_release():
    with pytest.raises(ValueError, match='container and live-provider evidence missing'):
        release_gate()


def test_required_control_cannot_be_removed():
    manifest = render()
    launch_policy(manifest)
    manifest['x-anchor-launch']['required_controls'].remove('kill_switch')
    with pytest.raises(ValueError, match='launch controls missing'):
        launch_policy(manifest)


def test_full_evaluation_is_required_even_after_report_validation(monkeypatch):
    monkeypatch.setattr(evaluation, 'require_evidence', lambda *a, **k: {'tier': 'sample'})
    with pytest.raises(ValueError, match='full_launch_evaluation_required'):
        evaluation.require_launch_evidence({})


def test_production_uses_accumulated_gate_before_infrastructure(monkeypatch):
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(launch.subprocess, 'run', runner)
    with pytest.raises(subprocess.CalledProcessError):
        launch.production()
    assert calls[0][0] == ['ruby', 'bin/release_gate']
    assert calls[0][1]['env']['ANCHOR_PRODUCTION_LAUNCH'] == '1'


def test_accumulated_gate_retains_product_and_quality_boundaries():
    source = (launch.ROOT / 'bin/release_gate').read_text()
    for required in ('"bin/rails", "test"', '"anchor_ai/tests"',
                     'eval/retrieval_gate.py', 'eval/rag_gate.py',
                     'eval/agent_gate.py', 'eval/multi_agent_gate.py',
                     'eval.redteam_gate', 'bin/redteam_harness.rb',
                     '"eval.release_gate", "decide"', 'require_launch_evidence'):
        assert required in source
