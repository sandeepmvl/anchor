import argparse
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPLAY = ROOT / 'eval/datasets/triad_replay_scores.json'
DIMENSIONS = ('context_relevance', 'groundedness', 'answer_relevance')
METRICS = {
    'ragas': {'ContextRelevance': 'context_relevance',
              'faithfulness': 'groundedness', 'answer_relevancy': 'answer_relevance'},
    'trulens': {name: name for name in DIMENSIONS},
}


def fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_score(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('invalid_triad_score')
    return float(value)


def normalize(framework, scores):
    if framework not in METRICS:
        raise ValueError('unknown_framework')
    mapping = METRICS[framework]
    if set(scores) != set(mapping):
        raise ValueError('unknown_or_missing_metric_key')
    mapped = {mapping[key]: validate_score(value) for key, value in scores.items()}
    return {dimension: mapped[dimension] for dimension in DIMENSIONS}


class RecordedTriadClient:
    def __init__(self, path=REPLAY):
        if os.getenv('ANCHOR_EVAL_PROVIDER', 'offline') != 'offline':
            raise ValueError('live_triad_adapter_not_implemented')
        self.recording = json.loads(Path(path).read_text(encoding='utf-8'))

    def scores(self, framework, row):
        if framework not in METRICS:
            raise ValueError('unknown_framework')
        key = row.get('id')
        case = self.recording['cases'].get(key)
        if case is None:
            raise ValueError('unknown_replay_key')
        expected = dict(self.recording['snapshots'][case['snapshot']],
                        id=key, category=case['category'])
        if row != expected:
            raise ValueError('stale_replay_input')
        return self.recording['profiles'][case['profile']][framework].copy()


def load_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()
            if line.strip()]
    ids = [row['id'] for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError('empty_batch_or_duplicate_id')
    return rows


def evaluate(framework, rows, client=None):
    client = client or RecordedTriadClient()
    ids = [row['id'] for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError('empty_batch_or_duplicate_id')
    results = []
    for row in rows:
        raw = client.scores(framework, row)
        results.append({
            'id': row['id'], 'tenant_id': row['tenant_id'],
            'generation_id': row['generation_id'], 'category': row['category'],
            'input_fingerprint': fingerprint(row),
            'scores': normalize(framework, raw), 'raw_scores': raw,
        })
    recording = client.recording
    return {
        'schema_version': 1, 'framework': framework, 'mode': 'offline',
        'provenance': recording['provenance'],
        'configuration': recording['configuration'],
        'replay_fingerprint': fingerprint(recording),
        'dataset_fingerprint': fingerprint(sorted(rows, key=lambda row: row['id'])),
        'metric_mapping': METRICS[framework], 'rows': results,
        'aggregates': {dimension: sum(row['scores'][dimension] for row in results) / len(results)
                       for dimension in DIMENSIONS},
    }


def print_report(report):
    print(f"{report['framework']}: OFFLINE SYNTHETIC REPLAY; {len(report['rows'])} cases")
    print('id | context_relevance | groundedness | answer_relevance')
    for row in report['rows']:
        print(row['id'] + ' | ' + ' | '.join(f"{row['scores'][d]:.3f}" for d in DIMENSIONS))
    print('MEAN | ' + ' | '.join(f"{report['aggregates'][d]:.3f}" for d in DIMENSIONS))


def compare_reports(left, right):
    for field in ('schema_version', 'mode', 'dataset_fingerprint', 'replay_fingerprint'):
        if left[field] != right[field]:
            raise ValueError('incomparable_reports')
    a = {row['id']: row for row in left['rows']}
    b = {row['id']: row for row in right['rows']}
    if set(a) != set(b):
        raise ValueError('different_query_sets')
    for key in sorted(a):
        if a[key]['input_fingerprint'] != b[key]['input_fingerprint']:
            raise ValueError('different_query_inputs')
    print('Comparable triad reports for the same evaluation tickets')
    print('dimension | RAGAS mean | TruLens mean')
    for dimension in DIMENSIONS:
        print(f"{dimension} | {left['aggregates'][dimension]:.3f} | "
              f"{right['aggregates'][dimension]:.3f}")
    low_left = {key for key, row in a.items() if row['scores']['context_relevance'] < 0.5}
    low_right = {key for key, row in b.items() if row['scores']['context_relevance'] < 0.5}
    print('RAGAS low context: ' + ', '.join(sorted(low_left)))
    print('TruLens low context: ' + ', '.join(sorted(low_right)))
    print('Replay validates report plumbing, not changes in live answer quality.')


def main(framework):
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-set', required=True)
    parser.add_argument('--output')
    parser.add_argument('--compare')
    args = parser.parse_args()
    report = evaluate(framework, load_rows(args.eval_set))
    print_report(report)
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    if args.compare:
        other = json.loads(Path(args.compare).read_text(encoding='utf-8'))
        if other['framework'] != 'ragas' or framework != 'trulens':
            raise ValueError('compare_requires_ragas_then_trulens')
        compare_reports(other, report)
