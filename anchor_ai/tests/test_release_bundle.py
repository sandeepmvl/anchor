from dataclasses import FrozenInstanceError
from pathlib import Path
import copy
import pytest
import yaml

from anchor_ai.serving_artifacts import pin_release_bundle
from anchor_ai.semantic_cache import SemanticCache, Generated, snapshot_for_release
from ops.deploy.release_bundle import identity
from eval.release_gate import release_bundle_fixture_gate, decide, manifest, offline_bundle

ROOT = Path(__file__).resolve().parents[2]


def snapshot(name='good', generation=0):
    bundle = yaml.safe_load((ROOT / 'config/model_releases.yml').read_text())['release_bundles'][name]
    return dict(tenant_id=1, generation=generation, bundle_id=identity(bundle), bundle=bundle)


def admit(value, offline=True):
    return pin_release_bundle(value, tenant_id=1, required_contract='support-tools:v1', offline=offline)


def scope(pinned):
    return snapshot_for_release(pinned, role='customer', audience='public', locale='en',
        permission_version='acl:v1', answer_path='faq', kb='public-kb:v1',
        embedding_version='hash:v1', eligibility='public_faq')


def test_request_pins_one_immutable_bundle_during_swap():
    source = snapshot('degraded', 1)
    in_flight = admit(source)
    source.update(snapshot('good', 2))
    restored = admit(source)
    assert in_flight.prompt == 'chat:rollback-degraded-v1'
    assert in_flight.complete(in_flight.ticket) == 'Order A17 has already been refunded.'
    assert restored.complete(restored.ticket) == 'Order A17 requires a policy check and confirmation before refund.'
    assert (restored.prompt, restored.model, restored.index) == ('chat:v3', 'default-recorded', 'support-index:v1')
    with pytest.raises(FrozenInstanceError):
        in_flight.prompt = restored.prompt
    assert restored.cache_namespace != in_flight.cache_namespace
    assert scope(in_flight).prompt_version == in_flight.prompt


def test_cache_never_mixes_candidate_restored_or_index_only_bundles():
    cache = SemanticCache(emit=lambda event: None)
    good, bad, old_index = [admit(snapshot(name)) for name in ('good', 'degraded', 'prior_index')]
    query = 'What is the public return policy?'
    vector = [1.0, 0.0]
    assert cache.store(scope(bad), query, vector, Generated('Check the published policy.', public_faq=True))
    assert cache.lookup(scope(good), query, vector) is None
    assert cache.lookup(scope(old_index), query, vector) is None
    assert cache.store(scope(good), query, vector, Generated('Read the current policy.', public_faq=True))
    assert cache.lookup(scope(admit(snapshot('good', 9))), query, vector) == 'Read the current policy.'
    with pytest.raises(ValueError, match='override'):
        snapshot_for_release(good, prompt_version=bad.prompt)


def test_missing_real_evidence_tampering_tenant_and_contract_fail_closed():
    with pytest.raises(ValueError, match='production_readiness'):
        admit(snapshot(), offline=False)
    value = snapshot()
    value['bundle']['model'] = 'unreviewed-model'
    with pytest.raises(ValueError, match='identity_changed'):
        admit(value)
    value = snapshot()
    value['tenant_id'] = 2
    with pytest.raises(ValueError, match='tenant_mismatch'):
        admit(value)
    value = copy.deepcopy(snapshot())
    value['bundle']['contract'] = 'removed-tool:v0'
    value['bundle_id'] = identity(value['bundle'])
    with pytest.raises(ValueError, match='roll_forward'):
        admit(value)


def test_golden_regression_gate_and_production_scope():
    assert release_bundle_fixture_gate(snapshot()['bundle'])['passed']
    bad = release_bundle_fixture_gate(snapshot('degraded')['bundle'])
    assert not bad['passed']
    assert bad['rows']['golden'] == 'failed'
    expected = manifest('release_bundle', snapshot()['bundle'])
    assert not decide(offline_bundle(expected), expected, production=True)['passed']
