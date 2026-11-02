import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.regression_provider import (ReplayClient, digest, envelope,
    fixture_input, load_manifest, REPLAYS)
from eval.regression_assertions import results


def incident_checks():
    import asyncio
    import tempfile
    from anchor_ai.runtime import incident_regression_cases, recovered_trace
    cases = incident_regression_cases()
    with tempfile.TemporaryDirectory() as directory:
        corrected = asyncio.run(recovered_trace(Path(directory) / 'corrected'))
        human = asyncio.run(recovered_trace(Path(directory) / 'human', human=True))
    if corrected['authority']['used'] > 12 or human['calls']:
        raise ValueError('failover_budget_incident_regression')
    return dict(passed=True, cases=cases,
                reservations=corrected['authority']['used'],
                human_only_provider_calls=len(human['calls']),
                deployment_approved=False,
                limitation='Offline incident replay; hosted and multi-host release evidence not verified')


def validate_input(source):
    manifest = source['manifest']
    if manifest != load_manifest():
        raise ValueError('manifest_hash_mismatch')
    if [p['tag'] for p in source['prompts']] != ['chat:v1', 'chat:v2']:
        raise ValueError('prompt_pair_mismatch')
    for prompt in source['prompts']:
        checked = envelope(prompt['contract'], prompt['version_id'])
        if checked != prompt:
            raise ValueError('prompt_hash_mismatch')
    models = {p['contract']['model_id'] for p in source['prompts']}
    if len(models) != 1:
        raise ValueError('model_mismatch')
    return manifest


def run(source, regressed=False):
    manifest = validate_input(source)
    client = ReplayClient()
    rows = []
    for prompt in source['prompts']:
        for example in manifest['examples']:
            for repeat in range(manifest['repeat']):
                answer = client.complete(prompt['contract'], example, regressed)
                rows.append(dict(tag=prompt['tag'], id=example['id'], repeat=repeat,
                    output=answer['text'], rubric_score=answer['rubric_score'],
                    rubric_source=answer['rubric_source'],
                    **results(answer['text'], manifest['expected'], manifest['similarity_threshold'])))
    return dict(format='anchor-regression-v2', source=source,
        input_hash=digest(source), manifest_hash=digest(manifest),
        replay_hash=digest(REPLAYS.read_text()), cache=False,
        provider_calls=client.calls, evidence='constructed_replay',
        cli_integration_verified=False, deployment_approved=False, rows=rows)


def gate(report, release=False):
    source = report['source']
    manifest = validate_input(source)
    if report['input_hash'] != digest(source) or report['manifest_hash'] != digest(manifest):
        raise ValueError('manifest_hash_mismatch')
    if report['replay_hash'] != digest(REPLAYS.read_text()):
        raise ValueError('replay_hash_mismatch')
    if report['cache'] is not False:
        raise ValueError('cached_repeats')
    expected = {(p['tag'], e['id'], r) for p in source['prompts']
                for e in manifest['examples'] for r in range(manifest['repeat'])}
    found = [(r['tag'], r['id'], r['repeat']) for r in report['rows']]
    if len(found) != len(expected) or set(found) != expected:
        raise ValueError('missing_or_duplicate_outputs')
    if report['provider_calls'] != len(expected):
        raise ValueError('provider_call_count_mismatch')
    checked = []
    for row in report['rows']:
        if not isinstance(row['output'], str) or not row['output'].strip():
            raise ValueError('missing_output')
        checked.append(dict(row, **results(row['output'], manifest['expected'], manifest['similarity_threshold'])))
    candidate = [r for r in checked if r['tag'] == 'chat:v2']
    rate = sum(sum(r['checks'].values()) for r in candidate) / (len(candidate) * 4)
    regressed = []
    unsafe = sorted({r['id'] for r in checked if not r['checks']['safety']})
    for example in manifest['examples']:
        rates = {}
        for tag in ('chat:v1', 'chat:v2'):
            group = [r for r in checked if r['id'] == example['id'] and r['tag'] == tag]
            rates[tag] = sum(all(r['checks'].values()) for r in group) / len(group)
        if (rates['chat:v1'] >= manifest['baseline_minimum'] and
                rates['chat:v1'] - rates['chat:v2'] > manifest['tolerance']):
            regressed.append(example['id'])
    blockers = []
    incident_checks()
    if rate < manifest['assertion_threshold']:
        blockers.append('candidate_assertion_pass_rate')
    if regressed:
        blockers.append('case_regression')
    if unsafe:
        blockers.append('non_negotiable_safety')
    if release:
        blockers.append('fresh_hosted_and_calibrated_judge_evidence_missing')
    return dict(passed=not blockers, assertion_pass_rate=rate,
        regressed_ids=regressed, unsafe_ids=unsafe, blockers=blockers,
        deployment_approved=False,
        limitation='Offline adapter smoke only; Promptfoo CLI and hosted quality not verified')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', default='tmp/regression_report.json')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--input')
    parser.add_argument('--regressed', action='store_true')
    parser.add_argument('--release', action='store_true')
    parser.add_argument('--incident-only', action='store_true')
    parser.add_argument('--expect-blocked', action='store_true',
                        help='Smoke assertion only; never use this option as a CI release gate')
    args = parser.parse_args()
    path = Path(args.report)
    try:
        if args.incident_only:
            decision = incident_checks()
            if args.release:
                decision.update(passed=False, blockers=['fresh_hosted_and_calibrated_judge_evidence_missing'])
            print(json.dumps(decision, indent=2))
            return 0 if decision['passed'] else 1
        if args.run:
            source = json.loads(Path(args.input).read_text()) if args.input else fixture_input()
            report = run(source, args.regressed)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2))
        else:
            report = json.loads(path.read_text())
        decision = gate(report, args.release)
    except (ValueError, KeyError, TypeError, OSError, AssertionError) as error:
        print(json.dumps({'passed': False, 'error': str(error)}))
        return 1
    print(json.dumps(decision, indent=2))
    print('Same 20 support questions; worsened answers: ' + (', '.join(decision['regressed_ids']) or 'none'))
    status = 0 if decision['passed'] else 1
    if args.expect_blocked:
        return 0 if status == 1 and decision['unsafe_ids'] == ['refund-01'] else 1
    return status


if __name__ == '__main__':
    sys.exit(main())
