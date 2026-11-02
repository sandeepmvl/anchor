import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anchor_ai.hosting_economics import compare, require_production_routing


def inputs():
    rows = [json.loads(line) for line in (ROOT / 'eval/fixtures/hosting_usage_30_days.jsonl').read_text().splitlines()]
    config = yaml.safe_load((ROOT / 'config/self_hosted.yml').read_text())['hosting_economics']
    return rows, config


def test_workload_choices_include_constraints_and_missing_evidence():
    report = compare(*inputs())
    assert report['observed_days'] == 30
    assert not report['production_routing_enabled']
    workloads = report['workloads']
    assert workloads['ticket_triage']['recommendation']['option'] == 'managed'
    assert workloads['kb_embedding']['recommendation']['option'] == 'self_hosted'
    triage_matrix = workloads['ticket_triage']['matrix']
    assert triage_matrix[0]['monthly_usd'] == 49.6
    assert triage_matrix[1]['monthly_usd'] == 33.6
    feasible_costs = {row['option']: row['monthly_usd']
                      for row in triage_matrix if row['planning_feasible']}
    assert feasible_costs['managed'] < feasible_costs['hosted']
    assert workloads['ticket_triage']['recommendation']['option'] == min(feasible_costs, key=feasible_costs.get)
    assert workloads['kb_embedding']['matrix'][2]['monthly_usd'] == 1000
    assert not workloads['ticket_triage']['matrix'][2]['planning_feasible']
    for data in workloads.values():
        assert len(data['matrix']) == 3
        assert data['policy']['recovery_limit_ms'] > 0
        for row in data['matrix']:
            assert row['latency_provenance'] == 'synthetic_not_measured'
            assert row['privacy_constraints'] and row['operations_burden']
            assert len(row['missing_evidence']) == 5


def test_idle_capacity_fully_loaded_cost_and_sensitivity():
    baseline = compare(*inputs())['workloads']['kb_embedding']
    low = compare(*inputs(), utilization=0.05, operations_usd=680)['workloads']['kb_embedding']
    assert baseline['capacity']['replicas'] == 1
    assert baseline['capacity']['idle_capacity_fraction'] > 0
    assert baseline['capacity']['utilization_assumption'] == 0.2
    assert baseline['capacity']['infrastructure_usd'] == 100
    assert baseline['capacity']['operations_usd'] == 180
    assert baseline['capacity']['headroom_fraction'] == 0.2
    assert low['capacity']['replicas'] == 2
    assert low['matrix'][2]['monthly_usd'] == 2220
    assert baseline['break_even_sensitivity'] != low['break_even_sensitivity']
    crossings = baseline['break_even_sensitivity']
    assert {r['competitor'] for r in crossings} == {'hosted', 'managed'}
    assert len({r['operations_usd'] for r in crossings}) == 3
    assert all('one_replica_capacity_million' in r and 'fits_one_replica' in r for r in crossings)


@pytest.mark.parametrize('claim', ['synthetic', 'live_approved'])
def test_production_routing_refuses_even_claimed_live_evidence(claim):
    report = compare(*inputs())
    report.update(basis=claim, production_routing_enabled=True)
    with pytest.raises(RuntimeError, match='validator missing'):
        require_production_routing(report)


def test_unknown_workload_and_invalid_utilization_fail_closed():
    rows, config = inputs()
    with pytest.raises(ValueError):
        compare(rows, config, utilization=0)
    with pytest.raises(ValueError, match='reviewed policy'):
        compare([dict(rows[0], workload='unknown')], config)
