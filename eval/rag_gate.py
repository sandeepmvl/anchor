"""Fresh offline pipeline evaluation and fail-closed release control."""
import argparse
import asyncio
import copy
import json
import os
import subprocess
import uuid
from pathlib import Path

from anchor_ai.rag import answer_release
from eval.heuristic_judge import METRIC_NAMES, score_capture, write_json
from eval.triad import fingerprint, validate_score

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / 'eval/datasets/rag_thresholds.json'


def load_dataset():
    return json.loads(DATASET.read_text())


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def offline_search(chunks, ticket, degraded=False):
    """Fake retrieval I/O only; generation and scoring still execute."""
    async def search(query):
        scoped = [copy.deepcopy(c) for c in chunks if
                  c['tenant_id'] == ticket['tenant_id'] and
                  c['generation_id'] == ticket['generation_id']]
        if degraded:
            return [c for c in scoped if c['id'] == 903]
        # The first lookup reveals a follow-up link; the second resolves it.
        return [c for c in scoped if c['id'] == (902 if query == 'refund approval policy'
                and query != ticket['question'] else 901)]
    return search


async def capture(dataset, provenance, client=None, degraded=False):
    rows = []
    for ticket in dataset['tickets']:
        rows.append(await answer_release(ticket, dataset['configuration'],
            offline_search(dataset['chunks'], ticket, degraded), client=client))
    return dict(provenance=provenance, rows=rows)


def validate_trajectories(captured, dataset):
    tickets = {ticket['id']: ticket for ticket in dataset['tickets']}
    rows = captured['rows']
    ids = [row['id'] for row in rows]
    if not tickets or len(tickets) != len(dataset['tickets']):
        raise ValueError('invalid_golden_tickets')
    if len(ids) != len(tickets) or set(ids) != set(tickets):
        raise ValueError('partial_duplicate_or_unexpected_capture')
    for row in rows:
        ticket = tickets[row['id']]
        canonical = {c['id'] for c in dataset['chunks'] if
                     c['tenant_id'] == ticket['tenant_id'] and
                     c['generation_id'] == ticket['generation_id']}
        if (row['pipeline'] != 'advanced' or row['tenant_id'] != ticket['tenant_id'] or
                row['generation_id'] != ticket['generation_id'] or
                row['question'] != ticket['question']):
            raise ValueError('wrong_pipeline_or_scope')
        events = row['trajectory']
        calls = [event for event in events if event['event'] == 'retrieve']
        if (not events or events[-1]['event'] != 'stop' or
                events[-1]['outcome'] not in ('complete', 'budget') or
                events[-1]['tool_calls'] != len(calls) or
                not 1 <= len(calls) <= dataset['configuration']['max_tool_calls']):
            raise ValueError('invalid_bounded_trajectory')
        if len(calls) != ticket['expected_calls']:
            raise ValueError('missing_multihop_coverage')
        if sum(event['event'] == 'rerank' for event in events) != 1:
            raise ValueError('missing_reranker')
        observed = set()
        for event in events:
            evidence_ids = event.get('evidence_ids', [])
            if (any(type(i) is not int for i in evidence_ids) or
                    set(evidence_ids) - canonical):
                raise ValueError('invented_trajectory_evidence')
            observed.update(evidence_ids)
        packed = {c['id'] for c in row['evidence']}
        if packed - observed or set(row['citations']) - packed:
            raise ValueError('invalid_capture_evidence')
        if not set(ticket['required_ids']) <= observed:
            raise ValueError('missing_expected_evidence')


def enforce(scores, captured, dataset, provenance):
    if dataset['configuration']['advanced_rag'] is not True:
        raise ValueError('advanced_flag_required')
    if captured['provenance'] != provenance or scores['provenance'] != provenance:
        raise ValueError('stale_provenance')
    if provenance['config_hash'] != fingerprint(dataset['configuration']):
        raise ValueError('stale_configuration')
    if provenance['dataset_hash'] != fingerprint(dataset):
        raise ValueError('stale_dataset')
    expected_ids = [t['id'] for t in dataset['tickets']]
    actual_ids = [row['id'] for row in scores['rows']]
    if len(actual_ids) != len(expected_ids) or set(actual_ids) != set(expected_ids):
        raise ValueError('partial_duplicate_or_unexpected_scores')
    for row in scores['rows']:
        if set(row['scores']) != set(METRIC_NAMES):
            raise ValueError('missing_or_unknown_metric')
        for value in row['scores'].values():
            validate_score(value)
    # Equality includes capture hashes, aggregates, adapter mappings and row hashes.
    if scores != score_capture(captured):
        raise ValueError('scores_do_not_match_current_capture')
    decisions = {}
    for metric in METRIC_NAMES:
        value = validate_score(scores['aggregates'][metric])
        baseline = validate_score(dataset['baseline'][metric])
        tolerance = validate_score(dataset['tolerance'][metric])
        floor = max(0.0, baseline - tolerance)
        decisions[metric] = dict(score=value, baseline=baseline, tolerance=tolerance,
                                 floor=floor, passed=value >= floor)
    # Keep metric failures visible even when retrieval also broke a trajectory.
    if not all(row['passed'] for row in decisions.values()):
        return decisions, 'metric_regression'
    validate_trajectories(captured, dataset)
    return decisions, None


def run_gate(output, dataset=None, client=None, degraded=False, release=None):
    dataset = copy.deepcopy(dataset or load_dataset())
    output = Path(output)
    report = dict(passed=False, metrics={}, error=None)
    try:
        if os.getenv('ANCHOR_LLM_PROVIDER', 'offline') != 'offline':
            raise ValueError('offline_gate_requires_offline_provider')
        if os.getenv('ANCHOR_EVAL_PROVIDER', 'offline') != 'offline':
            raise ValueError('offline_gate_requires_offline_judge')
        if dataset['configuration']['provider'] != 'offline':
            raise ValueError('unsupported_evaluation_provider')
        sha = git('rev-parse', 'HEAD')
        if release:
            if git('status', '--porcelain'):
                raise ValueError('release_requires_clean_checkout')
            subprocess.run(['git', 'check-ref-format', 'refs/tags/' + release],
                           cwd=ROOT, check=True, capture_output=True)
            existing = subprocess.run(['git', 'rev-parse', '--verify', 'refs/tags/' + release],
                                      cwd=ROOT, capture_output=True)
            if existing.returncode == 0:
                raise ValueError('release_tag_already_exists')
        provenance = dict(run_id=uuid.uuid4().hex, commit_sha=sha,
            config_hash=fingerprint(dataset['configuration']),
            dataset_hash=fingerprint(dataset), flag=dataset['configuration']['advanced_rag'])
        report.update(provenance)
        report['configuration'] = dataset['configuration']
        report['dataset_version'] = dataset['version']
        captured = asyncio.run(capture(dataset, provenance, client, degraded))
        write_json(output / 'capture.json', captured)
        scores = score_capture(captured)
        write_json(output / 'scores.json', scores)
        # Exercise the persisted boundary, not an in-memory success variable.
        persisted = json.loads((output / 'scores.json').read_text())
        decisions, error = enforce(persisted, captured, dataset, provenance)
        report.update(metrics=decisions, error=error, passed=error is None,
                      capture_hash=fingerprint(captured), score_hash=fingerprint(scores))
        if release and (git('rev-parse', 'HEAD') != sha or git('status', '--porcelain')):
            report.update(passed=False, error='source_changed_during_evaluation')
        write_json(output / 'report.json', report)
        if release and report['passed']:
            subprocess.run(['git', '-c', 'user.name=Anchor Release', '-c',
                'user.email=release@anchor.invalid', 'tag', '-a', release, sha, '-m',
                f"Anchor RAG evidence {output / 'report.json'} sha256:{fingerprint(report)}"],
                cwd=ROOT, check=True, capture_output=True)
    except Exception as exc:
        report.update(passed=False, error=type(exc).__name__ + ':' + str(exc))
    write_json(output / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output')
    parser.add_argument('--release')
    args = parser.parse_args()
    default = Path(git('rev-parse', '--git-path', 'anchor-eval')) / uuid.uuid4().hex
    output = Path(args.output) if args.output else default
    report = run_gate(output, release=args.release)
    print(json.dumps(report, sort_keys=True))
    print(('PASS' if report['passed'] else 'BLOCK') + ': fresh offline RAG; report=' +
          str(output / 'report.json'))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
