"""Snapshot-bound calibration. Recorded evidence is never release approval."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jsonschema
import yaml
from anchor_ai.llm_client import CalibrationClient
from eval.heuristic_judge import score_with_rubric, write_json

LABELS = ('Fail', 'Pass')


def digest(value):
    data = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def index(rows):
    result = {}
    for row in rows:
        key = row['id']
        if not isinstance(key, str) or not key or key in result:
            raise ValueError('duplicate_or_invalid_id')
        result[key] = row
    if not result:
        raise ValueError('empty_rows')
    return result


def snapshot_hash(rows):
    indexed = index(rows)
    return digest([indexed[key] for key in sorted(indexed)])


def bind_labels(rows, labels):
    examples, references = index(rows), index(labels)
    if set(examples) != set(references):
        raise ValueError('missing_or_extra_labels')
    schema = json.loads((ROOT / 'eval/golden_schema.json').read_text())['$defs']['calibrationLabel']
    for label in labels:
        jsonschema.validate(label, schema)
        if label['snapshot_example']['id'] != label['id']:
            raise ValueError('label_identity_mismatch')
    actual = snapshot_hash(rows)
    reference_hash = snapshot_hash([label['snapshot_example'] for label in labels])
    if actual != reference_hash:
        raise ValueError('snapshot_hash_mismatch')
    return actual, {(reference_hash, label['id']): label for label in labels}


def categorical_metrics(humans, predictions):
    if not humans or len(humans) != len(predictions):
        raise ValueError('missing_labels')
    matrix = [[0, 0], [0, 0]]
    for human, prediction in zip(humans, predictions):
        if human not in LABELS or prediction not in LABELS:
            raise ValueError('invalid_label')
        matrix[LABELS.index(human)][LABELS.index(prediction)] += 1
    n = len(humans)
    matches = matrix[0][0] + matrix[1][1]
    expected = sum(sum(matrix[i]) * sum(matrix[j][i] for j in range(2)) for i in range(2))
    denominator = n * n - expected
    if denominator == 0:
        raise ValueError('degenerate_kappa')
    return dict(n=n, agreement=matches / n,
                kappa=(matches * n - expected) / denominator,
                confusion=matrix, order=list(LABELS), rows='human', columns='judge')


def selected_rubric(document, revision):
    if revision not in document['revisions']:
        raise ValueError('unknown_rubric_revision')
    selected = {key: value for key, value in document.items() if key != 'revisions'}
    selected.update(document['revisions'][revision])
    return selected


def load_inputs():
    rows = read_jsonl(ROOT / 'labs/llm-judge-calibration/golden_v2.jsonl')
    labels = read_jsonl(ROOT / 'eval/datasets/judge_human_labels_v2.jsonl')
    policy = json.loads((ROOT / 'eval/judge_calibration_policy.json').read_text())
    rubric = yaml.safe_load((ROOT / 'eval/judge_rubric.yml').read_text())
    return rows, labels, policy, rubric


def validate_splits(rows, policy):
    if any(row['split'] not in {'dev', 'holdout'} for row in rows):
        raise ValueError('invalid_split')
    for split in ('dev', 'holdout'):
        if sum(row['split'] == split for row in rows) < policy['minimum_examples_per_split']:
            raise ValueError('insufficient_split')
    holdout_tags = {tag for row in rows if row['split'] == 'holdout' for tag in row['tags']}
    if set(policy['required_holdout_tags']) - holdout_tags:
        raise ValueError('missing_holdout_tags')


def summarize(records):
    result = categorical_metrics([r['human'] for r in records], [r['judge']['label'] for r in records])
    tags = sorted({tag for row in records for tag in row['tags']})
    result['per_tag'] = {}
    for tag in tags:
        selected = [row for row in records if tag in row['tags']]
        # Small or single-class tags still have meaningful raw agreement.
        result['per_tag'][tag] = dict(n=len(selected), agreement=sum(
            row['human'] == row['judge']['label'] for row in selected) / len(selected))
    return result


def evaluate(rows, labels, policy, document, revision='baseline', split='dev', client=None):
    if split not in {'dev', 'holdout'}:
        raise ValueError('invalid_split')
    snapshot, references = bind_labels(rows, labels)
    validate_splits(rows, policy)
    rubric = selected_rubric(document, revision)
    client = client or CalibrationClient()
    # Recordings are bound to exactly the committed prompt, not arbitrary edits.
    if client.mode == 'offline':
        committed = yaml.safe_load((ROOT / 'eval/judge_rubric.yml').read_text())
        if selected_rubric(committed, revision) != rubric:
            raise ValueError('unrecorded_rubric')
    records = []
    predictions = {}
    for row in sorted(rows, key=lambda item: item['id']):
        if row['split'] != split:
            continue
        key = (snapshot, row['id'])
        prediction = score_with_rubric(row, client, rubric, revision)
        predictions[key] = prediction
    selected = [row for row in rows if row['split'] == split]
    expected_keys = {(snapshot, row['id']) for row in selected}
    if set(predictions) != expected_keys:
        raise ValueError('missing_judge_labels')
    for row in selected:
        key = (snapshot, row['id'])
        records.append(dict(id=row['id'], tags=row['tags'],
                            human=references[key]['label'], judge=predictions[key]))
    metrics = summarize(records)
    critical = sum(row['human'] == 'Fail' and row['judge']['label'] == 'Pass'
                   and bool(set(row['tags']) & set(policy['critical_tags'])) for row in records)
    checks = dict(holdout_evaluated=split == 'holdout',
                  kappa_met=metrics['kappa'] >= policy['minimum_holdout_kappa'],
                  critical_met=critical <= policy['maximum_critical_false_passes'])
    report = dict(schema_version=1, dataset='golden_v2', snapshot_hash=snapshot,
                  labels_hash=digest(sorted(labels, key=lambda row: row['id'])),
                  rubric_hash=digest(rubric), rubric_version=rubric['version'],
                  policy_hash=digest(policy), revision=revision, split=split,
                  judge=dict(mode=client.mode, model='recorded-calibration-v1' if client.mode == 'offline'
                             else os.getenv('OPENAI_MODEL', 'gpt-4o-mini'), max_tokens=1024,
                             decoding='recorded' if client.mode == 'offline' else 'provider_default'),
                  label_origins=sorted({label['origin'] for label in labels}),
                  metrics=metrics, critical_false_passes=critical,
                  checks=checks, policy_met=all(checks.values()), production_approved=False,
                  limitation='Recorded responses and synthetic labels do not verify live model agreement or injection robustness.')
    if split == 'dev':
        report['disagreements'] = [row for row in records if row['human'] != row['judge']['label']]
        report['rows'] = records
    return report


def require_release_approval(report, policy):
    # No production evidence ingestion/attestation exists in this increment.
    # A hand-edited report or production_enabled=true cannot enable release.
    raise ValueError('production_judge_not_approved: independent human/live evidence and release integration required')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['golden_v2'])
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--revision', choices=['baseline', 'clarified'], default='baseline')
    parser.add_argument('--split', choices=['dev', 'holdout'], default='dev')
    parser.add_argument('--compare', action='store_true')
    args = parser.parse_args()
    rows, labels, policy, document = load_inputs()
    client = CalibrationClient(offline=True if args.offline else None)
    report = evaluate(rows, labels, policy, document, args.revision, args.split, client)
    if args.compare:
        if args.split != 'dev':
            raise ValueError('comparison_is_development_only')
        baseline = evaluate(rows, labels, policy, document, 'baseline', 'dev', client)
        report['development_kappa_change'] = report['metrics']['kappa'] - baseline['metrics']['kappa']
    path = ROOT / 'tmp/judge_calibration' / f'{args.revision}-{args.split}.json'
    write_json(path, report)
    print(json.dumps(report, indent=2, allow_nan=False))
    print('Report:', path.relative_to(ROOT))
    print('Production influence BLOCKED: synthetic/recorded evidence is not approval.')


if __name__ == '__main__':
    main()
