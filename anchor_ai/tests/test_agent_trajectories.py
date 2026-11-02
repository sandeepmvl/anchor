import asyncio
import copy

import pytest

from eval.rag_gate import capture, load_dataset, validate_trajectories


def golden_capture():
    dataset = load_dataset()
    return dataset, asyncio.run(capture(dataset, {}))


def test_golden_trajectories_enforce_bounded_multihop_coverage():
    dataset, captured = golden_capture()
    validate_trajectories(captured, dataset)
    for row in captured['rows']:
        calls = [e for e in row['trajectory'] if e['event'] == 'retrieve']
        assert len(calls) == 2
        assert calls[0]['query'] == row['question']
        assert calls[1]['query'] == 'refund approval policy'
        assert 902 in calls[1]['evidence_ids']
        assert row['trajectory'][-1]['outcome'] == 'complete'
        assert row['pipeline'] == 'advanced'


@pytest.mark.parametrize('fault', ['budget', 'identifier', 'hop', 'stop', 'coverage'])
def test_trajectory_faults_are_rejected(fault):
    dataset, captured = golden_capture()
    events = captured['rows'][0]['trajectory']
    if fault == 'budget':
        events.insert(0, copy.deepcopy(events[0]))
        events[-1]['tool_calls'] = 3
    elif fault == 'identifier':
        events[0]['evidence_ids'].append(999999)
    elif fault == 'hop':
        del events[1]
        events[-1]['tool_calls'] = 1
    elif fault == 'stop':
        events.pop()
    else:
        for event in events:
            if 'evidence_ids' in event:
                event['evidence_ids'] = [i for i in event['evidence_ids'] if i != 901]
    with pytest.raises(ValueError):
        validate_trajectories(captured, dataset)


def test_unknown_stop_outcome_is_rejected():
    dataset, captured = golden_capture()
    captured['rows'][0]['trajectory'][-1]['outcome'] = 'unknown'
    with pytest.raises(ValueError):
        validate_trajectories(captured, dataset)


@pytest.mark.parametrize('strategy', ['react', 'plan_execute'])
def test_bounded_strategies_preserve_retrieval_and_stop_safely(tmp_path, strategy):
    from anchor_ai.tests.test_bounded_runtime import scenario, run
    runtime, backend = scenario.make_runtime(tmp_path / 'normal')
    runtime.policy['bounded']['strategies']['multi-step refund'] = strategy
    paused = run(runtime)
    assert paused['strategy'] == strategy
    assert paused['memory'][0]['observation'] == asyncio.run(
        scenario.offline_retrieve(scenario.identities()['worker'], paused['message']))
    assert [h['id'] for h in paused['memory'][0]['observation']['hits']] == [901, 902]
    runtime.review('reviewer', 'refund-104', False)
    rejected = run(runtime, operation='resume')
    assert rejected['status'] == 'rejected'
    assert rejected['escalation']['reason'] == 'approval_rejected'
    assert backend.count() == 0

    runtime, backend = scenario.make_runtime(tmp_path / 'denied')
    runtime.policy['bounded']['strategies']['multi-step refund'] = strategy
    denied = run(runtime, token='reader')
    assert denied['status'] == 'denied'
    assert denied['attempts'][-1]['allowed'] is False
    assert denied['escalation']['attempts'][-1]['action'] == 'issue_refund'
    assert backend.count() == 0

    runtime, backend = scenario.make_runtime(tmp_path / 'exhausted')
    runtime.policy['bounded']['strategies']['multi-step refund'] = strategy
    runtime.policy['run']['max_tokens'] = 7
    exhausted = run(runtime)
    assert exhausted['status'] == 'exhausted'
    assert exhausted['steps'] == exhausted['tokens'] == 0
    assert not any(e['subsystem'] == 'planner' for e in exhausted['events'])
    assert backend.count() == 0


def test_single_lookup_routes_to_react(tmp_path):
    from anchor_ai.tests.test_bounded_runtime import scenario, run
    runtime, backend = scenario.make_runtime(tmp_path)
    done = run(runtime, ticket_type='single-lookup')
    assert done['strategy'] == 'react'
    assert done['status'] == 'completed'
    assert [m['action'] for m in done['memory']] == ['retrieve']
    assert backend.count() == 0


@pytest.mark.parametrize('fault', ['unauthorized', 'tenant-leak', 'duplicate', 'missing-escalation'])
def test_agent_gate_rejects_corrupted_replay(tmp_path, fault):
    from eval.agent_gate import capture_case, validate, corrupt, load_cases
    case = load_cases()[0]
    trace = asyncio.run(capture_case(case, tmp_path))
    # The legacy replay has approval but no proven dry-run or recovery capability.
    with pytest.raises(ValueError, match='unsafe_trajectory:.*confirmation_and_recovery_required'):
        validate(trace, case)
    expected_error = {
        'unauthorized': 'unauthorized_mutation',
        'tenant-leak': 'tenant_memory_leak',
        'duplicate': 'duplicate_or_missing_side_effect',
        'missing-escalation': 'missing_required_escalation',
    }[fault]
    # Require the original guard's error, not merely any trajectory rejection.
    with pytest.raises(ValueError, match='^' + expected_error + '$'):
        validate(corrupt(trace, fault), case)
