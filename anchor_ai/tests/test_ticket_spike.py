import asyncio
import pytest
from ops.load.ticket_spike import (
    Admission, band, configuration, first_knee, require_capacity_evidence, summarize)
from anchor_ai.observability import spike_measurement


def test_failure_accounting_does_not_improve_success_percentiles():
    rows = [dict(outcome='success', started=1, latency_ms=100,
                 queue_ms=20, generation_ms=80),
            dict(outcome='provider_429', started=1, latency_ms=1,
                 queue_ms=0, generation_ms=0),
            dict(outcome='timeout', started=1, latency_ms=200,
                 queue_ms=None, generation_ms=None),
            dict(outcome='admission_rejected', started=0, latency_ms=0,
                 queue_ms=None, generation_ms=None)]
    result = summarize(rows, 1, 4, 'test')
    assert result['scheduled'] == result['completed'] + result['failures'] == 4
    assert result['started'] == 3
    assert result['successful_latency_ms']['p99'] == 100
    assert result['successful_req_s'] == 1
    assert result['outcomes']['timeout'] == 1
    assert result['outcomes']['provider_429'] == 1


def test_provider_throttling_and_worker_wait_are_distinct():
    c = configuration()
    result = asyncio.run(band(16, c, every_429=4))
    assert result['outcomes']['provider_429'] == 25
    assert result['worker_queue_ms']['p99'] > 0
    assert result['completed'] == 75
    assert result['failures'] == 25


def test_timeout_is_counted_and_cancellation_releases_workers():
    result = asyncio.run(band(16, configuration(), timeout_s=.000001))
    assert result['outcomes'].get('timeout', 0) > 0
    assert result['scheduled'] == result['completed'] + result['failures']
    assert result['started'] == 100


def test_knee_uses_first_tested_objective_miss_with_queue_evidence():
    def sample(n, latency, queue):
        return dict(concurrency=n, successful_latency_ms={'p99': latency},
                    worker_queue_ms={'p99': queue}, failures=7)
    c = dict(p99_objective_ms=100, queue_objective_ms=20)
    bands = [sample(1, 40, 0), sample(4, 110, 0), sample(16, 150, 30),
             sample(32, 200, 80)]
    assert first_knee(bands, c) == 16
    assert bands[2]['failures'] == 7
    assert first_knee(bands[:2], c) is None


def test_tenant_pending_rate_and_cost_limits_are_independent():
    c = configuration()
    c.update(tenant_pending=1, tenant_requests=2, tenant_cost_units=2)
    p = Admission(c)
    assert p.enter(1) is None
    assert p.enter(1) == 'admission_rejected'
    assert p.enter(2) is None
    p.leave(1)
    assert p.enter(1) is None
    p.leave(1)
    assert p.enter(1) == 'rate_rejected'
    c.update(tenant_requests=10, tenant_cost_units=1)
    p = Admission(c)
    assert p.enter(1) is None
    p.leave(1)
    assert p.enter(1) == 'cost_rejected'
    assert p.enter(2) is None


def test_offline_cannot_release_capacity():
    with pytest.raises(ValueError, match='live_capacity_evidence_required'):
        require_capacity_evidence({'execution_mode': 'offline'})
    with pytest.raises(ValueError, match='live_capacity_evidence_required'):
        require_capacity_evidence({'execution_mode': 'live'})


def test_telemetry_rejects_content_and_invalid_durations():
    row = dict(tenant_id=1, trace_id='1' * 32, started=1, outcome='timeout',
               latency_ms=10, queue_ms=None, generation_ms=None)
    assert spike_measurement(row) == row
    with pytest.raises(ValueError):
        spike_measurement(dict(row, message='private ticket'))
    with pytest.raises(ValueError):
        spike_measurement(dict(row, latency_ms=float('nan')))
