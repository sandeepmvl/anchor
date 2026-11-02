import argparse
import json
from pathlib import Path

from anchor_ai.pii import redact_evaluation
from eval.faithfulness import ROOT, make_judge, persist_report, score_payload


def load_dataset():
    return json.loads((ROOT / 'eval/datasets/groundedness_cases.json').read_text())


def evaluate_cases(dataset=None, judge=None):
    dataset = dataset or load_dataset()
    judge = judge or make_judge()
    reports = []
    for case in dataset['cases']:
        payload = dict(answer=case['answer'], citations=case['citations'],
            evidence=dataset['evidence'], tenant_id=dataset['tenant_id'],
            generation_id=dataset['generation_id'])
        reports.append(dict(case=case['id'], **score_payload(payload, judge)))
    return redact_evaluation(dict(dataset_version=dataset['version'],
        query=dataset['query'], judge=type(judge.client).__name__, reports=reports))


def run_gate():
    dataset = load_dataset()
    # CI always replays recordings, regardless of a developer's live setting.
    from eval.faithfulness import LLMJudge, RecordedFaithfulnessClient
    report = evaluate_cases(dataset, LLMJudge(RecordedFaithfulnessClient()))
    for case, result in zip(dataset['cases'], report['reports']):
        expected_status = 'no_claims' if case['expected_score'] is None else 'scored'
        if result['status'] != expected_status or result['score'] != case['expected_score']:
            raise ValueError('faithfulness_regression:' + case['id'])
    print('PASS: offline citation-aware faithfulness gate; synthetic recordings')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', required=True)
    parser.add_argument('--report', help='Write a redacted report')
    parser.add_argument('--show-calls', action='store_true')
    args = parser.parse_args()
    dataset = load_dataset()
    if args.query != dataset['query']:
        parser.error('The recording contains only the refund-policy query')
    judge = make_judge()
    report = evaluate_cases(dataset, judge)
    print(json.dumps(report, indent=2))
    if args.show_calls:
        print(json.dumps(redact_evaluation(getattr(judge.client, 'calls', [])), indent=2))
    if args.report:
        persist_report(Path(args.report), report)


if __name__ == '__main__':
    main()
