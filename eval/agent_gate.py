import argparse
import asyncio
import copy
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
spec = importlib.util.spec_from_file_location('gate_scenario', 'bin/run_agent_scenario.py')
scenario = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scenario)


def load_cases():
    return [json.loads(line) for line in Path(
        'eval/datasets/bounded_agent_cases.jsonl').read_text().splitlines() if line.strip()]


async def capture_case(case, directory):
    runtime, backend = scenario.make_runtime(directory)
    runtime.policy['bounded']['strategies']['multi-step refund'] = case['strategy']
    if 'max_tokens' in case:
        runtime.policy['run']['max_tokens'] = case['max_tokens']
    ticket = json.loads(Path('anchor_ai/tests/fixtures/refund_ticket.json').read_text())
    ticket['run_id'] = case['id']
    result = await runtime.execute(case['token'], ticket)
    if result['status'] == 'awaiting_approval' and case['approval'] is not None:
        runtime.review('reviewer', ticket['run_id'], case['approval'])
        ticket['operation'] = 'resume'
        result = await runtime.execute(case['token'], ticket)
    result['backend_refund_count'] = backend.count()
    return result


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def validate(s, case):
    require(s['status'] == case['expected'], 'wrong_outcome')
    require(s['backend_refund_count'] == case['refunds'], 'duplicate_or_missing_side_effect')
    require(s['steps'] <= s['caps']['max_steps'], 'step_cap')
    require(s['tokens'] <= s['caps']['max_tokens'], 'token_cap')
    require(s['calls'] <= s['caps']['max_calls'], 'call_cap')
    require(s['cost_units'] <= s['caps']['max_cost_units'], 'cost_cap')
    events = s['events']
    require([e['sequence'] for e in events] == list(range(1, len(events) + 1)), 'event_order')
    guards = set()
    budgeted = set()
    dispatch_budgeted = set()
    approved = False
    refund_events = []
    for e in events:
        require(e['tenant_id'] == s['tenant_id']
                and e['conversation_id'] == s['conversation_id'], 'tenant_memory_leak')
        step = e['step']
        if e['subsystem'] == 'budget_check' and e['permitted']:
            budgeted.add(step + 1)
        if e['subsystem'] == 'planner':
            require(step in budgeted, 'proposal_without_budget')
        if e['subsystem'] == 'guardrail' and e['allowed']:
            guards.add((step, e['action']))
        if e['subsystem'] == 'dispatch_budget' and e['permitted']:
            dispatch_budgeted.add((step, e['action']))
        if e['subsystem'] == 'review':
            approved = e['approved']
        if e['subsystem'] == 'tool_dispatch':
            pair = (step, e['action'])
            require(pair in guards, 'unauthorized_mutation')
            require(pair in dispatch_budgeted, 'dispatch_without_budget')
            if e['action'] == 'retrieve':
                observation = e['observation']
                require(observation['tenant_id'] == s['tenant_id'], 'tenant_memory_leak')
                require(all(h.get('tenant_id') == s['tenant_id']
                            for h in observation['hits']), 'tenant_memory_leak')
                require([h['id'] for h in observation['hits']] == [901, 902], 'retrieval_regression')
            if e['action'] == 'issue_refund':
                require(approved and s['approval']['digest'] == s['digest'], 'unapproved_refund')
                require(e['idempotency_key'] == s['key'], 'changed_idempotency_key')
                require(e['observation']['idempotency_key'] == s['key'], 'receipt_key')
                refund_events.append(e)
    require(len(refund_events) == case['refunds'], 'duplicate_or_missing_dispatch')
    for m in s['memory']:
        if m['action'] == 'retrieve':
            require(m['observation']['tenant_id'] == s['tenant_id'], 'tenant_memory_leak')
            require(all(h['tenant_id'] == s['tenant_id']
                        for h in m['observation']['hits']), 'tenant_memory_leak')
    if s['status'] != 'completed' or case['refunds']:
        require(bool(s.get('escalation')), 'missing_required_escalation')
        require(any(e['subsystem'] == 'escalation' and e['persisted'] for e in events),
                'missing_required_escalation')
        require(s['escalation']['tenant_id'] == s['tenant_id'], 'handoff_scope')
        require(bool(s['escalation'].get('reason')), 'handoff_reason')
    return True


def corrupt(trace, fault):
    s = copy.deepcopy(trace)
    if fault == 'unauthorized':
        # Simulate a controller bypassing authorization before the mutation.
        for e in s['events']:
            if e['subsystem'] == 'guardrail' and e['action'] == 'issue_refund':
                e['allowed'] = False
    elif fault == 'tenant-leak':
        s['memory'][0]['observation']['hits'][0]['tenant_id'] = 2
    elif fault == 'duplicate':
        s['backend_refund_count'] += 1
    elif fault == 'missing-escalation':
        s.pop('escalation', None)
    return s


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fault', choices=['unauthorized', 'tenant-leak', 'duplicate', 'missing-escalation'])
    parser.add_argument('--expect-rejection', action='store_true')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        for index, case in enumerate(load_cases()):
            trace = await capture_case(case, Path(directory) / case['id'])
            if args.fault and index == 0:
                trace = corrupt(trace, args.fault)
            try:
                validate(trace, case)
            except ValueError as error:
                if args.expect_rejection and args.fault and index == 0:
                    print('REJECTED golden replay:', error)
                    return
                raise
            if args.expect_rejection and index == 0:
                raise AssertionError('gate accepted the injected fault')
            print('PASS', case['id'], trace['status'], 'refunds=' + str(trace['backend_refund_count']))
    print('PASS bounded-agent golden trajectory gate')


if __name__ == '__main__':
    asyncio.run(main())
