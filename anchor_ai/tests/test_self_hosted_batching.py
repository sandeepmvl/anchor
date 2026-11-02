import copy
from types import SimpleNamespace

import pytest

@pytest.fixture(autouse=True)
def isolated_benchmark_dependencies(monkeypatch):
    """Do not export offline benchmark configuration to quota tests."""
    import importlib
    import os

    monkeypatch.setattr(os, 'environ', os.environ.copy())
    benchmark = importlib.import_module('anchor_ai.serving_benchmark')
    monkeypatch.setitem(globals(), 'bench', benchmark)


def test_profiles_and_deployment_arguments():
    for name in ('default', 'tuned'):
        _, profile, _, kv = bench.surge_profile(name)
        assert kv > 0
        command = bench.surge_launch(name)
        for key, value in profile.items():
            flag = '--' + key.replace('_', '-')
            assert command[command.index(flag) + 1] == str(value)
        assert '--enable-chunked-prefill' in command


@pytest.mark.parametrize('key,value', [
    ('max_num_seqs', 0), ('max_num_seqs', 65),
    ('max_num_seqs', True), ('max_num_batched_tokens', 2),
    ('max_num_batched_tokens', 4097),
])
def test_invalid_bounds_fail_closed(monkeypatch, key, value):
    config = copy.deepcopy(bench.configuration())
    config['batching_profiles']['default'][key] = value
    monkeypatch.setattr(bench, 'configuration', lambda: config)
    with pytest.raises(ValueError, match='invalid_batching_bounds'):
        bench.surge_profile('default')


def test_identical_surge_compares_successes_not_offered_rate(monkeypatch):
    def forbidden_wall_clock():
        raise AssertionError('surge must use virtual service time')
    # Replace only the benchmark's clock reference. HTTPX may time protocol
    # exchanges internally, but those timings must never enter surge reports.
    monkeypatch.setattr(bench, 'time', SimpleNamespace(perf_counter=forbidden_wall_clock))
    before = bench.surge(50, 'default')
    after = bench.surge(50, 'tuned')
    assert before == bench.surge(50, 'default')
    assert after == bench.surge(50, 'tuned')
    assert before['workload_sha256'] == after['workload_sha256']
    assert after['completed_req_s'] > before['completed_req_s']
    assert after['latency_p99_s'] != before['latency_p99_s']
    # Healthy profiles may complete every request. Force admission pressure
    # separately so failed-work accounting is tested without requiring failure
    # from a healthy default or tuned profile.
    pressured = bench.surge(50, 'tuned', admission=dict(
        tenant_inflight_limit=3, tenant_token_quota=500, request_deadline_s=1))
    assert pressured['rejected'] > 0
    assert pressured['failures'] > 0
    assert pressured['completed'] < pressured['samples']
    assert (pressured['completed_req_s']
            < pressured['samples'] / pressured['duration_virtual_s'])
    for report in (before, after, pressured):
        assert report['completed'] + report['failures'] == report['samples']
        assert report['failures'] == report['timeouts'] + report['rejected']
        assert report['completed_req_s'] == report['completed'] / report['duration_virtual_s']
        assert 'Real hardware measurements' in report['evidence_warning']
        assert report['production_promotion_allowed'] is False


def test_tenant_admission_is_preserved_under_concurrent_traffic():
    policy = dict(tenant_inflight_limit=3, tenant_token_quota=500, request_deadline_s=1)
    report = bench.surge(50, 'tuned', admission=policy)
    for tenant in report['tenants'].values():
        assert tenant['accepted'] > 0
        assert tenant['rejected'] > 0
        assert tenant['peak_inflight'] <= 3
        assert tenant['reserved_tokens'] <= 500
        assert tenant['inflight'] == 0
        assert tenant['offered'] == tenant['accepted'] + tenant['rejected']
        assert tenant['accepted'] == tenant['completed'] + tenant['timeouts']


def test_iteration_refill_beats_fixed_group_for_short_work():
    continuous = bench.surge(3, 'default', samples=8)
    static = bench.surge(3, 'default', 'static', samples=8)
    def admission(report, number):
        return next(row['time_s'] for row in report['trace'] if row['id'] == number)
    assert admission(continuous, 3) < admission(static, 3)


def test_missing_or_forged_hardware_evidence_cannot_promote():
    for report in ({}, {'execution_mode': 'live', 'gpu_identity': 'claimed'}):
        with pytest.raises(ValueError, match='real_hardware_measurements'):
            bench.require_surge_promotion(report)
