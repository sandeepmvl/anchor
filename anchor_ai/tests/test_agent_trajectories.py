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
