import copy
import time

import pytest

from eval.release_gate import CHECKS, decide, manifest, offline_bundle, policy, require_evidence, save


def test_approved_and_regressed_refund_controls():
    expected = manifest()
    assert decide(offline_bundle(expected), expected, production=False)['passed']
    verdict = decide(offline_bundle(expected, regressed=True), expected, production=False)
    assert not verdict['passed']
    assert verdict['rows']['trajectory'] == 'failed'
    assert verdict['rows']['regression'] == 'failed'


@pytest.mark.parametrize('name', CHECKS)
@pytest.mark.parametrize('fault', ['missing', 'failed', 'stale', 'mismatch'])
def test_every_report_is_required(name, fault):
    expected = manifest()
    bundle = offline_bundle(expected)
    if fault == 'missing':
        del bundle['reports'][name]
    elif fault == 'failed':
        bundle['reports'][name]['status'] = 'failed'
    elif fault == 'stale':
        bundle['reports'][name]['created_at'] = 0
    else:
        bundle['reports'][name]['manifest'] = dict(expected, prompt='different')
    assert not decide(bundle, expected, production=False)['passed']


@pytest.mark.parametrize('field', ['commit', 'dataset', 'rubric', 'prompt', 'model', 'index', 'release'])
def test_approval_binds_every_identity(field):
    expected = manifest()
    changed = dict(expected, **{field: 'different'})
    assert not decide(offline_bundle(expected), changed, production=False)['passed']


@pytest.mark.parametrize('state', ['failure', 'cancelled', 'skipped', None])
def test_bad_upstream_never_passes(state):
    expected = manifest()
    bundle = offline_bundle(expected)
    bundle['upstream']['trajectory'] = state
    assert not decide(bundle, expected, production=False)['passed']


def test_production_requires_real_scope_and_baseline(tmp_path):
    expected = manifest()
    evidence = save(tmp_path / 'offline.json', offline_bundle(expected))
    with pytest.raises(ValueError, match='evaluation_gate_failed'):
        require_evidence(expected, evidence)
    bundle = offline_bundle(expected)
    bundle['scope'] = 'production'
    assert 'baseline_required' in decide(bundle, expected)['errors']


def test_reviewed_quarantine_does_not_hide_missing_execution():
    expected = manifest()
    bundle = offline_bundle(expected)
    bundle['reports']['calibration']['score'] = 0
    config = copy.deepcopy(policy())
    config['quarantine']['calibration'] = dict(owner='eval-owner', reviewer='code-owner',
        issue='https://github.com/example/anchor/issues/3719', expires_at=time.time() + 3600)
    assert decide(bundle, expected, production=False, configuration=config)['rows']['calibration'] == 'quarantined'
    bundle['upstream']['calibration'] = 'skipped'
    assert not decide(bundle, expected, production=False, configuration=config)['passed']


def test_stale_evidence_refused_without_writing_selection(tmp_path):
    expected = manifest()
    evidence = save(tmp_path / 'stale.json', offline_bundle(expected, now=0))
    active = tmp_path / 'active.json'
    active.write_text('unchanged')
    with pytest.raises(ValueError, match='evaluation_gate_failed'):
        require_evidence(expected, evidence, offline=True)
    assert active.read_text() == 'unchanged'
