import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from anchor_ai import observability as obs
from anchor_ai.contracts import TriageLabels
from anchor_ai.llm_client import LlmClient
from anchor_ai.provider import ReplayProvider, REASONING_RESPONSES
from anchor_ai.reasoning import REPLAY_MODEL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--offline', action='store_true', required=True)
    parser.add_argument('--max-output-tokens', type=int, choices=[300, 4000], default=4000)
    parser.add_argument('--verify-policy', action='store_true')
    args = parser.parse_args()
    # Explicit providers make --offline win even over inherited live-mode variables.
    os.environ['OPENAI_MODEL'] = REPLAY_MODEL
    data = json.loads(REASONING_RESPONSES.read_text())
    records = []
    old_sink = obs.SINK
    old_model = None
    rows = []
    print('OFFLINE: authored fixtures; illustrative costs; fixture latency is not live latency')
    try:
        for ticket in data['tickets']:
            for effort in ('off', 'low', 'high'):
                if args.max_output_tokens == 300 and (ticket['id'], effort) != ('B1', 'high'):
                    continue
                provider = ReplayProvider(task='reasoning', fixture_id=ticket['id'],
                                          effort=effort, max_tokens=args.max_output_tokens)
                def capture(record):
                    record = dict(record, ticket_id=ticket['id'], ticket_class=ticket['class'])
                    records.append(record)
                obs.SINK = capture
                result = asyncio.run(LlmClient(provider).complete_result(
                    ticket['ticket'], max_tokens=args.max_output_tokens, effort=effort))
                correct = False
                if result.finish_reason == 'stop':
                    correct = TriageLabels.model_validate_json(result.reply).model_dump() == ticket['expected']
                record = records[-1]
                assert record['name'] == 'llm.call'
                rows.append(dict(ticket=ticket['id'], category=ticket['class'], effort=effort,
                                 correct=correct, reply=result.reply, record=record))
    finally:
        obs.SINK = old_sink
    output = Path(__file__).parent / 'reasoning_calls.jsonl'
    output.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in records))
    print('class setting correct incomplete reasoning fixture_latency_ms measured_replay_ms estimated_USD')
    for row in rows:
        r = row['record']
        print(row['category'], row['effort'], int(row['correct']),
              int(r['completion_status'] == 'incomplete'), r['tokens']['reasoning'],
              r['fixture_latency_ms'], r['latency_ms'], format(r['cost_estimate'], '.6f'))
    if args.max_output_tokens == 300:
        r = rows[0]['record']
        assert r['finish_reason'] == 'length' and r['completion_status'] == 'incomplete'
        assert r['tokens']['reasoning'] == r['tokens']['output'] == 300
        assert rows[0]['reply'] == '' and r['attempts'] == 1
        print('TRAP: empty answer, finish_reason=length, reasoning=output=cap=300; raise the cap')
    else:
        assert len(records) == 12
        for category in ('billing', 'technical', 'account', 'order'):
            group = [row['record'] for row in rows if row['category'] == category]
            assert all(r['completion_status'] == 'completed' for r in group)
            assert group[0]['tokens']['reasoning'] < group[1]['tokens']['reasoning'] < group[2]['tokens']['reasoning']
            assert group[0]['fixture_latency_ms'] < group[1]['fixture_latency_ms'] < group[2]['fixture_latency_ms']
        demo = [row for row in rows if row['ticket'] == 'B1']
        assert not demo[0]['correct'] and demo[-1]['correct']
        assert demo[0]['record']['cost_estimate'] < demo[-1]['record']['cost_estimate']
        print('DEMO: same B1 ticket, off -> high')
        for row in (demo[0], demo[-1]):
            r = row['record']
            print(row['effort'], row['reply'], 'reasoning_tokens=', r['tokens']['reasoning'],
                  'estimated_USD=', format(r['cost_estimate'], '.6f'),
                  'fixture_latency_ms=', r['fixture_latency_ms'])
        if args.verify_policy:
            import yaml
            policy = yaml.safe_load((Path(__file__).parent / 'thinking_budget.yml').read_text())
            expected = {'billing': 'low', 'technical': 'off', 'account': 'high', 'order': 'off'}
            assert policy['automatic_routing'] is False
            assert policy['status'] == 'suggested_offline_only'
            assert policy['max_completion_tokens'] == 4000
            assert {k: v['effort'] for k, v in policy['classes'].items()} == expected
            assert all(v['rationale'] for v in policy['classes'].values())
            print('POLICY:', expected, '(suggestion only; automatic routing disabled)')
    total = sum(r['cost_estimate'] for r in records)
    print('TOTAL:', len(records), 'offline calls; illustrative estimated USD', format(total, '.6f'))
    print('JSON log:', output.relative_to(ROOT))


if __name__ == '__main__':
    main()
