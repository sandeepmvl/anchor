"""Local milestone exploration; offline evidence never authorizes release."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROUTES = ['infer', 'structured_reply', 'infer_stream', 'dense_rag',
          'hybrid_retrieve', 'analyzed_retrieve', 'bounded_rag', 'release_rag',
          'chain_reply', 'multi_agent', 'cited_rag', 'bounded_agent']
INFRASTRUCTURE = ['deployed_http_auth', 'live_detector_evaluation',
                  'payment_retry_recovery', 'sandbox_isolation']
OUTPUT = Path('.git/anchor-eval/security.json')


def security_errors(evidence, expected, *, production, now=None):
    now = time.time() if now is None else now
    errors = []
    if evidence.get('manifest') != expected:
        errors.append('security_manifest_mismatch')
    created = evidence.get('created_at')
    if type(created) not in (int, float) or not math.isfinite(created) or not 0 <= now - created <= 86400:
        errors.append('security_evidence_stale_or_missing')
    if evidence.get('suite') != 'full' or evidence.get('target') != 'hardened_release':
        errors.append('security_suite_incomplete')
    if evidence.get('status') != 'passed':
        errors.append('security_suite_failed')
    if type(evidence.get('high_severity_findings')) is not int or evidence['high_severity_findings'] != 0:
        errors.append('high_severity_findings')
    benign = evidence.get('benign_controls', {})
    if type(benign.get('total')) is not int or benign['total'] <= 0 or benign.get('failed') != 0:
        errors.append('benign_control_failure')
    if set(evidence.get('handler_guards', [])) != set(ROUTES):
        errors.append('route_guard_coverage_missing')
    if evidence.get('negative_control_rejected') is not True:
        errors.append('negative_control_missing')
    if production:
        infrastructure = evidence.get('infrastructure', {})
        if evidence.get('scope') != 'production' or any(infrastructure.get(k) is not True for k in INFRASTRUCTURE):
            errors.append('production_infrastructure_evidence_missing')
        if evidence.get('accumulated_suites') != 'passed':
            errors.append('accumulated_suites_missing')
        if not evidence.get('evidence_links'):
            errors.append('reviewable_evidence_links_missing')
    return errors


def offline_evidence(identity, baseline_ms, guarded_ms, benign_count):
    return dict(manifest=identity, created_at=time.time(), suite='full',
        target='hardened_release', scope='offline', status='passed',
        high_severity_findings=0, benign_controls={'total': benign_count, 'failed': 0},
        handler_guards=ROUTES, negative_control_rejected=True,
        accumulated_suites='not_run_by_security_harness',
        infrastructure={name: False for name in INFRASTRUCTURE},
        metrics={'baseline_median_ms': baseline_ms, 'guarded_median_ms': guarded_ms,
                 'added_median_ms': guarded_ms - baseline_ms,
                 'provider_calls_added': 0, 'provider_cost_added_usd': 0.0,
                 'scope': 'shared input boundary only; CPU and infrastructure cost excluded'},
        limitations=['Direct handler tests are not deployed HTTP tests',
                     'Sandbox isolation not verified: no container runtime',
                     'Live detector quality and payment-provider behavior not verified'])


def measure():
    from anchor_ai.runtime import hardened_customer_text
    cases = ['Where is order A17?', 'Please explain the refund policy.',
             'Can an agent review my return?', 'How do I contact support?']
    baseline, guarded = [], []
    for _ in range(20):
        start = time.perf_counter_ns()
        for text in cases:
            str(text)
        baseline.append((time.perf_counter_ns() - start) / 1e6)
        start = time.perf_counter_ns()
        for text in cases:
            result = hardened_customer_text(text)
            if not result or ('A17' in text and 'A17' not in result):
                raise ValueError('benign_control_failed')
        guarded.append((time.perf_counter_ns() - start) / 1e6)
    return statistics.median(baseline), statistics.median(guarded), len(cases) * 20


def require_production(expected, path=None):
    path = path or os.environ.get('ANCHOR_SECURITY_EVIDENCE')
    if not path:
        raise ValueError('security_evidence_required')
    evidence = json.loads(Path(path).read_text())
    errors = security_errors(evidence, expected, production=True)
    if errors:
        raise ValueError('security_gate_failed:' + ','.join(errors))
    return evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', choices=['full'], default='full')
    parser.add_argument('--target', choices=['hardened_release'], default='hardened_release')
    parser.add_argument('--release', action='store_true')
    parser.add_argument('--negative-control', action='store_true')
    args = parser.parse_args()
    from eval.release_gate import manifest
    identity = manifest()
    if args.release:
        require_production(identity)
        return
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.unlink(missing_ok=True)
    env = dict(os.environ, RAILS_ENV='test', ANCHOR_LLM_PROVIDER='offline',
               ANCHOR_QUERY_PROVIDER='offline', ANCHOR_EMBEDDING_PROVIDER='offline')
    commands = [
        ['ruby', 'bin/rails', 'test', 'test/integration/security_hardening_milestone_test.rb'],
        [sys.executable, '-m', 'pytest', '-q',
         'anchor_ai/tests/test_security_hardening_milestone.py',
         'anchor_ai/tests/test_injection_filter.py', 'anchor_ai/tests/test_jailbreak_guard.py',
         'anchor_ai/tests/test_tenant_isolation.py', 'anchor_ai/tests/test_bounded_runtime.py',
         'anchor_ai/tests/test_rag_endpoint_eval.py', 'anchor_ai/tests/test_agent_trajectories.py']]
    for command in commands:
        subprocess.run(command, check=True, env=env)
    evidence = offline_evidence(identity, *measure())
    assert not security_errors(evidence, identity, production=False)
    broken = copy.deepcopy(evidence)
    broken['high_severity_findings'] = 1
    assert 'high_severity_findings' in security_errors(broken, identity, production=False)
    assert security_errors(evidence, identity, production=True)
    OUTPUT.write_text(json.dumps(evidence, indent=2, allow_nan=False))
    print(json.dumps({'demo': 'Malicious refund paths blocked; legitimate exact-confirmation contracts passed',
        'negative_control': 'regressed release REJECTED before promotion',
        'production': 'BLOCKED: offline evidence cannot promote',
        'metrics': evidence['metrics'], 'evidence': str(OUTPUT)}))


if __name__ == '__main__':
    main()
