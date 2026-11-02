import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
import io

ROOT = Path(__file__).resolve().parents[1]
CHECKS = ('golden', 'calibration', 'trajectory', 'regression')
WORK = ROOT / '.git/anchor-eval'


def policy():
    return json.loads((ROOT / 'config/eval_gate.yml').read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def manifest(kind='application', target=None):
    result = {}
    for name, relative in policy()['inputs'].items():
        result[name] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    process = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                             capture_output=True, text=True)
    result['commit'] = process.stdout.strip() if process.returncode == 0 else 'uncommitted'
    result['release'] = digest({'kind': kind, 'target': target})
    return result


def valid_time(value, now, age):
    return (type(value) in (int, float) and math.isfinite(value)
            and 0 <= now - value <= age)


def validate_reports(bundle, expected, now, max_age, scope):
    errors = []
    if bundle.get('manifest') != expected:
        errors.append('manifest_mismatch')
    if bundle.get('scope') != scope:
        errors.append('scope_mismatch')
    if bundle.get('tier') not in ('sample', 'full'):
        errors.append('invalid_tier')
    if not valid_time(bundle.get('created_at'), now, max_age):
        errors.append('stale_evidence')
    upstream = bundle.get('upstream', {})
    if set(upstream) != set(policy()['required_upstream']):
        errors.append('incomplete_upstream')
    for name in policy()['required_upstream']:
        if upstream.get(name) != 'success':
            errors.append(name + ':upstream_not_success')
    reports = bundle.get('reports', {})
    if set(reports) != set(CHECKS):
        errors.append('incomplete_reports')
    for name in CHECKS:
        report = reports.get(name, {})
        if report.get('manifest') != expected:
            errors.append(name + ':manifest_mismatch')
        if not valid_time(report.get('created_at'), now, max_age):
            errors.append(name + ':stale_report')
        if report.get('status') not in ('passed', 'failed'):
            errors.append(name + ':invalid_status')
        score = report.get('score')
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            errors.append(name + ':invalid_score')
        if type(report.get('step')) is not int or report['step'] < 0:
            errors.append(name + ':invalid_step')
        if not isinstance(report.get('example_id'), str):
            errors.append(name + ':missing_example')
    return errors


def decide(bundle, expected, production=True, now=None, configuration=None):
    now = time.time() if now is None else now
    config = policy() if configuration is None else configuration
    scope = 'production' if production else 'offline'
    errors = validate_reports(bundle, expected, now, config['max_age_seconds'], scope)
    if production and expected.get('commit') == 'uncommitted':
        errors.append('commit_required')
    baseline = bundle.get('baseline')
    if production:
        if not isinstance(baseline, dict):
            errors.append('baseline_required')
        else:
            identity = baseline.get('manifest', {})
            errors += ['baseline:' + error for error in validate_reports(
                baseline, identity, now, config['baseline_max_age_seconds'], 'production')]
            if (baseline.get('tier') != 'full' or baseline.get('source', {}).get('branch') != 'main'
                    or baseline.get('source', {}).get('conclusion') != 'success'
                    or baseline.get('source', {}).get('workflow') != '.github/workflows/eval-gate.yml'
                    or not baseline.get('source', {}).get('run_id')):
                errors.append('baseline_source_invalid')
            for field in ('dataset', 'rubric'):
                if identity.get(field) != expected.get(field):
                    errors.append('baseline:' + field + '_mismatch')
    rows = {}
    for name in CHECKS:
        report = bundle.get('reports', {}).get(name, {})
        state = 'failed'
        score = report.get('score')
        minimum = config['floors'][name]
        if production and isinstance(baseline, dict) and name in ('calibration', 'trajectory'):
            previous = baseline.get('reports', {}).get(name, {}).get('score')
            if type(previous) in (int, float) and math.isfinite(previous):
                minimum = max(minimum, previous - config['tolerance'])
        structurally_valid = not any(error.startswith(name + ':') for error in errors)
        passed = (structurally_valid and report.get('status') == 'passed'
                  and type(score) in (int, float) and score >= minimum)
        exception = config['quarantine'].get(name)
        if exception:
            reviewed = (name in ('calibration', 'trajectory')
                        and isinstance(exception, dict)
                        and exception.get('owner') and exception.get('reviewer')
                        and exception.get('owner') != exception.get('reviewer')
                        and str(exception.get('issue', '')).startswith('https://github.com/')
                        and valid_time(exception.get('expires_at'), now + 604800, 604800))
            if not reviewed:
                errors.append(name + ':invalid_quarantine')
            elif structurally_valid:
                state = 'quarantined'
        if state != 'quarantined':
            state = 'passed' if passed else 'failed'
            if not passed:
                errors.append(name + ':score_or_status_failed')
        rows[name] = state
    for name in config['quarantine']:
        if name not in CHECKS:
            errors.append('unknown_quarantine')
    return {'passed': not errors, 'errors': sorted(set(errors)), 'rows': rows}


def require_evidence(expected, evidence=None, offline=False):
    path = evidence or os.environ.get('ANCHOR_EVAL_EVIDENCE')
    if not path:
        raise ValueError('evaluation_evidence_required')
    try:
        bundle = json.loads(Path(path).read_text())
        verdict = decide(bundle, expected, production=not offline)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ValueError('evaluation_evidence_invalid') from exc
    if not verdict['passed']:
        raise ValueError('evaluation_gate_failed:' + ','.join(verdict['errors']))
    return bundle


def offline_bundle(expected, regressed=False, now=None):
    """Explicit endpoint fakes, never used by the production adapter boundary."""
    now = time.time() if now is None else now
    response = {'answer': 'refund', 'steps': ['refund']} if regressed else {
        'answer': 'escalate', 'steps': ['escalate']}
    safe = response['answer'] == 'escalate' and 'refund' not in response['steps']
    reports = {}
    for name in CHECKS:
        passed = name in ('golden', 'calibration') or safe
        reports[name] = dict(manifest=expected, created_at=now,
                             status='passed' if passed else 'failed',
                             score=float(passed), example_id='refund-unconfirmed', step=0)
    return dict(manifest=expected, created_at=now, scope='offline', tier='sample',
                upstream={key: 'success' for key in policy()['required_upstream']},
                reports=reports)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False))
    return path


def find_baseline():
    repo = os.environ['GITHUB_REPOSITORY']
    def api(path):
        result = subprocess.run(['gh', 'api', '--paginate', '--slurp', path],
                                check=True, capture_output=True, timeout=60)
        return json.loads(result.stdout)
    artifacts = [a for page in api(f'repos/{repo}/actions/artifacts?name=anchor-baseline&per_page=100')
                 for a in page['artifacts']]
    for artifact in sorted(artifacts, key=lambda a: a['created_at'], reverse=True):
        if artifact['expired']:
            continue
        run = api(f'repos/{repo}/actions/runs/{artifact["workflow_run"]["id"]}')[0]
        if (run['head_branch'] != 'main' or run['conclusion'] != 'success'
                or run['path'] != '.github/workflows/eval-gate.yml'):
            continue
        data = subprocess.run(['gh', 'api', f'repos/{repo}/actions/artifacts/{artifact["id"]}/zip'],
                              check=True, capture_output=True, timeout=60).stdout
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            baseline = json.loads(archive.read('baseline.json'))
        if baseline.get('tier') != 'full':
            continue
        baseline['source'] = dict(branch='main', conclusion='success',
                                  workflow=run['path'], run_id=str(run['id']))
        save(WORK / 'baseline.json', baseline)
        return
    raise ValueError('no_usable_main_baseline')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['manifest', 'candidate', *CHECKS, 'decide', 'baseline', 'smoke'])
    parser.add_argument('--regressed', action='store_true')
    parser.add_argument('--expect-blocked', action='store_true')
    parser.add_argument('--evidence', type=Path)
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    if args.command == 'manifest':
        save(WORK / 'manifest.json', manifest())
        return
    if args.command == 'baseline':
        find_baseline()
        return
    if args.command == 'smoke':
        bundle = offline_bundle(manifest(), args.regressed)
        if args.evidence:
            bundle = json.loads(args.evidence.read_text())
        verdict = decide(bundle, manifest(), production=False)
        save(WORK / 'evidence.json', bundle)
        from eval.dashboard import render
        text = render(bundle, verdict)
        (WORK / 'dashboard.md').write_text(text)
        print(text)
        print('OFFLINE ONLY: production judge, service DB, sandbox isolation and merge ruleset not verified.')
        if args.expect_blocked:
            if verdict['passed']:
                raise SystemExit('expected a blocked gate')
            print('Observed nonzero release verdict; expected failure captured by smoke harness.')
            return
        raise SystemExit(0 if verdict['passed'] else 1)
    if args.command in ('candidate', *CHECKS):
        if not os.environ.get('OPENAI_API_KEY', '').strip():
            raise ValueError('missing_llm_key')
        adapter = os.environ.get('ANCHOR_EVAL_ADAPTER')
        if not adapter or not Path(adapter).is_file():
            raise ValueError('production_composition_adapter_missing')
        identity = manifest()
        save(WORK / 'manifest.json', identity)
        subprocess.run([adapter, args.command, str(WORK / 'manifest.json'),
                        str(WORK / (args.command + '.json')), os.environ.get('EVAL_TIER', 'full')],
                       check=True, timeout=300)
        return
    needs = json.loads(os.environ.get('NEEDS_JSON', '{}'))
    identity = manifest()
    reports = {}
    for name in CHECKS:
        path = WORK / (name + '.json')
        if path.exists():
            try:
                reports[name] = json.loads(path.read_text())
            except ValueError:
                reports[name] = {}
    bundle = dict(manifest=identity, created_at=time.time(), scope='production',
                  tier=os.environ.get('EVAL_TIER', 'full'), reports=reports,
                  upstream={name: value['result'] for name, value in needs.items()})
    baseline_path = WORK / 'baseline.json'
    if baseline_path.exists():
        bundle['baseline'] = json.loads(baseline_path.read_text())
    verdict = decide(bundle, identity)
    from labs.security_hardening.milestone import require_production
    try:
        bundle['security'] = require_production(identity)
    except (OSError, ValueError, TypeError, KeyError) as error:
        verdict['passed'] = False
        verdict['errors'].append('security_evidence_missing_or_failed')
        verdict['rows']['security'] = 'failed'
    else:
        verdict['rows']['security'] = 'passed'
    save(WORK / 'evidence.json', bundle)
    from eval.dashboard import render
    summary = render(bundle, verdict)
    (WORK / 'dashboard.md').write_text(summary)
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as stream:
            stream.write(summary)
    print(summary)
    if verdict['passed'] and bundle['tier'] == 'full':
        bundle.pop('baseline', None)
        save(WORK / 'baseline.json', bundle)
    raise SystemExit(0 if verdict['passed'] else 1)


if __name__ == '__main__':
    main()
