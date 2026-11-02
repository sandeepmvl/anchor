import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile

import httpx

from anchor_ai.provider import OfflineDeployment, ProviderError
from anchor_ai.resilience import CircuitStore, ResilienceBoundary, configuration
from anchor_ai.provider import RoutingProvider
from anchor_ai.vllm_backend import VllmBackend, validate_deployment


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class ServingProfiles:
    """Single-host atomic pointer; profiles themselves are immutable snapshots."""
    def __init__(self, directory, profiles, initial='previous'):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.profiles = copy.deepcopy(profiles)
        self.pointer = self.directory / 'serving-profile.json'
        if not self.pointer.exists():
            self.select(initial)

    def select(self, name):
        if name not in self.profiles:
            raise ValueError('unknown_serving_profile')
        fd, temporary = tempfile.mkstemp(dir=self.directory)
        try:
            with os.fdopen(fd, 'w') as stream:
                json.dump({'profile': name}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.pointer)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def pin(self):
        name = json.loads(self.pointer.read_text())['profile']
        return name, copy.deepcopy(self.profiles[name])


class ReleaseRouter(RoutingProvider):
    def __init__(self, *args, workload, **kwargs):
        self.release_workload = workload
        super().__init__(*args, **kwargs)


class ReleaseService:
    """Generation only: never executes tools or replays committed actions."""
    def __init__(self, profiles, router_factory):
        self.profiles, self.router_factory = profiles, router_factory
        self.cache = {}
        self.human_only = False

    async def complete(self, request, context, *, workload='chat', configure=None):
        from anchor_ai.escalation import inference_handoff
        name, config = self.profiles.pin()
        if configure:
            configure(config)  # Trusted operator/test hook, never HTTP input.
        if self.human_only:
            return inference_handoff(context, 'human_only')
        release = config['self_hosted_release']
        rule = release.get('tenants', {}).get(context['tenant_id'], {}).get(workload, {})
        if release.get('human_only') is not False or rule.get('mode') == 'human':
            return inference_handoff(context, 'human_only')
        # Include privacy, readiness, tenant, conversation and committed actions.
        key = fingerprint([name, config, workload, context,
                           request.content.decode('utf-8')])
        if key in self.cache:
            return copy.deepcopy(self.cache[key])
        router = self.router_factory(name, config, workload)
        try:
            response = await router.route(request, context['tenant_id'])
            if response.status_code != 200:
                return inference_handoff(context, 'provider_rejected')
        except (ProviderError, TimeoutError):
            return inference_handoff(context, 'no_permitted_provider')
        result = {'status': 'answered', 'profile': name,
                  'provider': router.served_by,
                  'reply': response.json()['choices'][0]['message']['content'],
                  'trace_id': context['trace_id'], 'tool_calls': []}
        self.cache[key] = copy.deepcopy(result)
        return result


def build_profiles():
    import yaml
    from anchor_ai.provider import routing_configuration
    base = routing_configuration()
    spec = yaml.safe_load(Path('config/self_hosted.yml').read_text())['release_profiles']
    profiles = {}
    for name, version in spec['profiles'].items():
        c = copy.deepcopy(base)
        c['serving_versions'] = version
        c['order'] = ['local', 'primary'] if version['preferred'] == 'local' else ['primary', 'local']
        c['deployments'] = {n: copy.deepcopy(base['deployments']['primary']) for n in c['order']}
        for n, deployment in c['deployments'].items():
            deployment.update(ready=True, workloads=['chat'])
            if n == 'local':
                deployment.update(backend='self_hosted', model=version['model'])
        c['tenants'] = {1: dict(approved=c['order'], regions=['eu']),
                        2: dict(approved=['local'], regions=['eu']),
                        3: dict(approved=[], regions=['eu'])}
        mode = version['preferred']
        c['self_hosted_release'] = dict(version=spec['version'], human_only=False, tenants={
            1: {'chat': dict(mode=mode, approved=c['order'], external_processing=True)},
            2: {'chat': dict(mode='local', approved=['local'], external_processing=False)},
            3: {'chat': dict(mode='human', approved=[], external_processing=False)}})
        profiles[name] = c
    return profiles


def require_production(evidence=None):
    # No unreviewed JSON claim can authorize GPU or privacy deployment.
    raise ValueError('self_hosted_real_infrastructure_validator_missing')


class OfflineHarness:
    """No network, GPU, Rails database, real quota billing, or tool execution."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.now = 0.0
        self.local = VllmBackend(mode='offline', timeout=.1)
        self.hosted = OfflineDeployment()
        self.events, self.charges, self.committed_actions = [], set(), {'refund-R42'}
        self.profiles = ServingProfiles(self.directory, build_profiles())
        self.service = ReleaseService(self.profiles, self.factory)
        self.boundaries = {}

    def factory(self, name, config, workload):
        harness = self
        class FixtureRouter(ReleaseRouter):
            async def physical(self, destination, request, payload):
                # Explicit fixture-only replacement for Rails admission/accounting.
                # The production RoutingProvider.physical remains unchanged.
                operation = request.headers.get('x-request-id', 'recovery-probe')
                harness.charges.add(operation)
                harness.events.append(dict(trace=operation, provider=destination))
                mapped = dict(payload, model=self.config['deployments'][destination]['model'])
                try:
                    return await self.adapters[destination].send(request, mapped)
                except httpx.TransportError:
                    raise ProviderError('provider_transport_error') from None
        router = FixtureRouter({'local': self.local, 'primary': self.hosted}, None,
                               config=copy.deepcopy(config), workload=workload)
        p = configuration()
        p.update(p['self_hosted_release'])
        boundary = ResilienceBoundary(router, CircuitStore(
            self.directory / 'circuits.sqlite3', p, clock=lambda: self.now),
            p, clock=lambda: self.now, jitter=lambda: 0)
        router.route = boundary.route
        router.probe = boundary.probe
        self.boundaries[name] = boundary
        return router

    def context(self, tenant):
        return dict(tenant_id=tenant, conversation_id='ticket-42',
                    messages=[{'role': 'user', 'content': 'Where is order 42? me@example.com'}],
                    evidence=['kb:refund-policy'], plan_position='confirmation_generation',
                    remaining_budget={'attempts': 2}, committed_actions=['refund-R42'],
                    trace_id='ticket-42')

    async def chat(self, tenant=1, *, configure=None, workload='chat'):
        request = httpx.Request('POST', 'https://offline.invalid/v1/chat/completions',
            headers={'x-request-id': 'ticket-42'}, json=dict(model='support-answer',
            messages=self.context(tenant)['messages'], stream=False))
        return await self.service.complete(request, self.context(tenant),
                                           configure=configure, workload=workload)


async def demo():
    validate_deployment()
    with tempfile.TemporaryDirectory() as directory:
        h = OfflineHarness(directory)
        h.profiles.select('local-v1')
        healthy = await h.chat()
        h.service.cache.clear()
        h.local.server.stopped = True
        outage = await h.chat()
        local_only = await h.chat(2)
        h.service.human_only = True
        before = len(h.events)
        human_only = await h.chat()
        assert len(h.events) == before
        h.service.human_only = False
        h.profiles.select('previous')
        rollback = await h.chat()
        result = dict(healthy=healthy, outage=outage, local_only=local_only,
                      human_only=human_only, rollback=rollback,
                      fixture_charge_count=len(h.charges),
                      committed_actions=sorted(h.committed_actions), traces=h.events,
                      limitations='Offline protocol and fixture ledger only; GPU, model quality, Rails endpoint and real quotas not verified.')
        assert healthy['provider'] == 'local' and outage['provider'] == 'primary'
        assert local_only['human_handoff'] and human_only['human_handoff']
        assert rollback['profile'] == 'previous'
        return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['demo', 'gate'])
    args = parser.parse_args()
    if args.command == 'demo':
        print(json.dumps(asyncio.run(demo()), indent=2))
    else:
        try:
            require_production()
        except ValueError as error:
            print('BLOCKED:', error)
        else:
            raise AssertionError('missing evidence unexpectedly released')


if __name__ == '__main__':
    main()
