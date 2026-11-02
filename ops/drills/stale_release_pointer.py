"""Disposable Anchor mechanism drill; no deployed HTTP or live model evidence."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anchor_ai.semantic_cache import SemanticCache, Snapshot, Generated, with_answer_cache
from anchor_ai.resilience import CacheBypassBreaker, CircuitStore
from eval.release_gate import stale_release_probe

WORK = ROOT / 'tmp/stale-release'
QUESTION = 'Can I get a refund after 20 days?'
GOOD = 'Refunds are allowed within 30 days.'
WRONG = 'Refunds are allowed only within seven days.'
LIMITATION = ('OFFLINE ONLY: deployed HTTP, live provider/judge quality, distributed '
              'trace export and multi-worker breaker delivery not verified.')


def registry(command):
    process = subprocess.run(['ruby', 'labs/stale-release-pointer/registry.rb', command],
                             cwd=ROOT, text=True, capture_output=True, check=True)
    return json.loads(process.stdout)


def write(name, value):
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / name).write_text(json.dumps(value, indent=2))


def append(name, value):
    WORK.mkdir(parents=True, exist_ok=True)
    with (WORK / name).open('a') as stream:
        stream.write(json.dumps(value) + '\n')


def inject():
    event = registry('inject')
    write('control.json', {'cache_bypass': False})
    append('deploy.jsonl', event)
    print(json.dumps({'injected': event, 'liveness': registry('inspect')['liveness'],
                      'readiness': registry('readiness')}))
    print(LIMITATION)


class RecordedPolicyClient:
    def answer(self, pinned):
        # No real-provider switch exists in this diagnostic client. Production
        # uses Anchor's normal configured provider and its safety boundary.
        return pinned['bundle']['reply']


def snapshot(pinned):
    bundle = pinned['bundle']
    return Snapshot(tenant_id=1, role='customer', audience='public', locale='en',
                    permission_version='public:v1', answer_path='faq', kb='refund',
                    embedding_version='fixture:v1', prompt_version=bundle['prompt'],
                    model_version=bundle['model'], retrieval_version=bundle['index'],
                    eligibility='public_faq', release_namespace=pinned['cache_namespace'])


async def probe_case(use_cache):
    pinned = registry('pin')
    state = registry('inspect')
    active_id = pinned['active_bundle_id']
    active = state['bundles'][active_id]
    expected = dict(answer=GOOD, prompt_version=active['prompt'],
                    model_version=active['model'], index_version=active['index'],
                    registry_revision=active_id, cache_namespace=active_id,
                    active_bundle_id=active_id)
    snap = snapshot(pinned)
    events = []
    cache = SemanticCache(emit=events.append)
    # Rehydrate a recorded old entry under its ORIGINAL namespace, never retag it.
    bad_id, bad = next((key, value) for key, value in state['bundles'].items()
                       if value['reply'] == WRONG)
    old = dict(pinned, bundle=bad, cache_namespace=bad_id)
    if use_cache:
        cache.store(snapshot(old), QUESTION, [1., 0.], Generated(WRONG, public_faq=True))
    control = json.loads((WORK / 'control.json').read_text())
    breaker = CacheBypassBreaker()
    if control['cache_bypass']:
        breaker.trip(1, seconds=300, actor='drill-operator')
    trace_id = uuid.uuid4().hex
    parent = uuid.uuid4().hex[:16]
    calls = []

    async def generate():
        calls.append(True)
        for name, attributes in [
            ('retrieval', {'index_version': pinned['bundle']['index'], 'chunks': ['refund-30d']}),
            ('generation', {'model_version': pinned['bundle']['model'],
                            'prompt_version': pinned['bundle']['prompt']})]:
            append('traces.jsonl', dict(trace_id=trace_id, span_id=uuid.uuid4().hex[:16],
                                       parent_id=parent, span=name, **attributes))
        return Generated(RecordedPolicyClient().answer(pinned), public_faq=True), {
            'provider_called': True, 'evidence': 'deterministic_recorded_policy_client'}

    answer, evidence = await with_answer_cache(cache, snap, QUESTION, [1., 0.],
                                               generate, breaker=breaker)
    response = dict(answer=answer, http_status=200, **pinned['provenance'],
                    model_version=pinned['bundle']['model'],
                    index_version=pinned['bundle']['index'])
    verdict = stale_release_probe(response, expected, production=False)
    record = dict(trace_id=trace_id, span_id=parent, parent_id=None, span='conversation',
                  case='cached-refund' if use_cache else 'fresh-refund',
                  app_version='app-good', cache_hit=not bool(calls),
                  cache_bypass=control['cache_bypass'], response=response,
                  events=events, verdict=verdict)
    append('traces.jsonl', record)
    return record, expected


def evaluate(expect=None):
    records = [asyncio.run(probe_case(False))[0], asyncio.run(probe_case(True))[0]]
    passed = all(row['verdict']['passed'] for row in records)
    result = {'passed': passed, 'cases': records, 'deployment_approved': False}
    write('golden-result.json', result)
    print(json.dumps(result))
    print(LIMITATION)
    if expect is not None:
        if passed != (expect == 'pass'):
            raise SystemExit('unexpected golden result')
        print('Checkpoint satisfied; golden verdict remains ' + ('PASS' if passed else 'FAIL'))
    elif not passed:
        raise SystemExit(1)
    return result


def resync(bypass):
    ack = registry('resync')
    write('control.json', {'cache_bypass': bypass})
    append('deploy.jsonl', {'action': 'resync', 'ack': ack, 'cache_bypass': bypass,
                           'provider_safety_breakers_changed': False})
    print(json.dumps({'registry_ack': ack, 'cache_bypass_ack': bypass,
                      'scope': 'disposable_single_worker',
                      'provider_safety_breakers_changed': False}))


def smoke():
    inject()
    assert registry('inspect')['liveness'] is True
    assert registry('readiness')['consistent'] is False
    failed = evaluate('fail')
    assert failed['cases'][0]['cache_hit'] is False
    assert failed['cases'][1]['cache_hit'] is True
    assert 'prompt_version:mismatch' in failed['cases'][0]['verdict']['errors']
    resync(True)
    fixed = evaluate('pass')
    assert all(not case['cache_hit'] and case['cache_bypass'] for case in fixed['cases'])
    assert registry('readiness')['consistent'] is True
    response, expected = asyncio.run(probe_case(False))
    assert not stale_release_probe(response['response'], expected)['passed']
    assert registry('production')['production_blocked']
    now = [0.]
    breaker = CacheBypassBreaker(clock=lambda: now[0])
    breaker.trip(1, seconds=5, actor='test')
    assert breaker.bypass(1) and not breaker.bypass(2)
    now[0] = 5.
    assert not breaker.bypass(1)
    policy = {'trip_threshold': 1, 'cooldown_seconds': 30,
              'half_open_probes': 1, 'probe_lease_seconds': 10, 'reset_threshold': 1}
    circuits = CircuitStore(WORK / 'provider.sqlite3', policy)
    permit = circuits.acquire('fixture-provider')
    if permit is not None:
        circuits.finish('fixture-provider', permit, 'health_failure')
    before = circuits.snapshot('fixture-provider')
    breaker.trip(1, seconds=5, actor='test')
    assert circuits.acquire('fixture-provider') is None
    assert circuits.snapshot('fixture-provider') == before
    print('PASS: green liveness; failed semantic probe; pointer resync + cache bypass '
          'restored 30-day answer and provenance; provider safety circuit stayed open.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['inject', 'eval', 'inspect', 'resync', 'smoke'])
    parser.add_argument('--expect', choices=['pass', 'fail'])
    parser.add_argument('--cache-bypass', action='store_true')
    args = parser.parse_args()
    if args.command == 'inject':
        inject()
    elif args.command == 'eval':
        evaluate(args.expect)
    elif args.command == 'resync':
        resync(args.cache_bypass)
    elif args.command == 'smoke':
        smoke()
    else:
        print(json.dumps(registry('inspect'), indent=2))
        for name in ('deploy.jsonl', 'traces.jsonl'):
            path = WORK / name
            print(name + '\n' + (path.read_text() if path.exists() else '(none)'))


if __name__ == '__main__':
    main()
