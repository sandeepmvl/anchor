from contextlib import contextmanager


@contextmanager
def verified_routing_scope(router, *, verify, token, client_for, feature):
    """Verify once, then bind accounting and admission to the same identity."""
    if not token:
        raise ContextError('missing_execution_context')
    identity = verify(token)
    if (identity is None or type(identity.tenant_id) is not int
            or identity.tenant_id <= 0 or not identity.role):
        raise ContextError('unverified_execution_context')
    model = router.config['deployments'][router.config['order'][0]]['model']
    from anchor_ai.quota_client import quota_scope
    with router.collector.scope(tenant_id=identity.tenant_id, feature=feature,
                                role='base_completion', model=model):
        with quota_scope(client_for(identity, feature), bound=router.config['bound']):
            yield identity


async def run_diagnostic_ticket(runtime, token, service, *, confirmed=False, dry_run=True):
    """Dedicated read-only diagnostic path; never a financial-tool dispatcher."""
    from anchor_ai.tools import register_sandbox
    scope = runtime.identity(token)
    registry = register_sandbox({}, service)
    proposal = {'tool': 'run_diagnostic', 'arguments': {'script_id': 'ticket_diagnostic'}}
    tool = registry[proposal['tool']]
    arguments = tool.arguments.model_validate(dict(proposal['arguments'], confirmed=confirmed,
                                                   dry_run=dry_run))
    with span('python.diagnostic_agent', stage='agent', tenant_id=scope.tenant_id,
              conversation_id=scope.conversation_id):
        result = await tool.run(scope=scope, **arguments.model_dump())
        return tool.result.model_validate(result).model_dump()


import asyncio
import copy
import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
import yaml
from anchor_ai.contracts import AgentRequest, VerifiedExecution
from anchor_ai.observability import span, trajectory_decision, model_call, CURRENT
from anchor_ai.orchestration import select_flow, worker_for, propose

ACTIONS = ['retrieve', 'lookup_order', 'check_refund', 'issue_refund']
TERMINAL = {'completed', 'denied', 'exhausted', 'rejected', 'failed'}


def hardened_customer_text(value):
    """Internal callers share the HTTP input boundary; text grants no authority."""
    from anchor_ai.injection_filter import screen_customer
    from anchor_ai.jailbreak_guard import evaluate
    from anchor_ai.pii import redact_value
    safe, flags = screen_customer(value)
    if flags or evaluate(value).refused:
        raise ValueError('unsafe_customer_text')
    return redact_value(safe)


def hardened_observation(value):
    """Reject instruction overrides before evidence enters durable agent memory."""
    from anchor_ai.pii import redact_value
    def check(item):
        if isinstance(item, str):
            hardened_customer_text(item)
        elif isinstance(item, dict):
            for child in item.values():
                check(child)
        elif isinstance(item, list):
            for child in item:
                check(child)
    check(value)
    return redact_value(value)


def argument_fingerprint(arguments):
    return hashlib.sha256(json.dumps(arguments, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


class ContextError(ValueError):
    pass


class BoundedRuntime:
    """Durable workflow; adapters do not own authorization or checkpoints.

    model.propose(strategy, plan, memory, max_tokens) -> (action, actual_tokens)
    retrieve(scope, message) -> scoped evidence dictionary
    backend.call(scope, action, arguments, key) -> authoritative observation
    Financial backends MUST durably deduplicate key and bind it to arguments.
    """
    def __init__(self, path, *, verify, model, retrieve, backend,
                 policy_path='config/agent_policy.yml', emit=None,
                 incident_check=None):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.verify = verify
        self.model = model
        self.retrieve = retrieve
        self.backend = backend
        self.incident_check = incident_check
        self.policy = yaml.safe_load(Path(policy_path).read_text())
        self.emit = emit or (lambda event: None)
        with self.db() as db:
            db.execute('CREATE TABLE IF NOT EXISTS bounded_runs ('
                       'tenant INTEGER, conversation INTEGER, run TEXT, state TEXT, '
                       'PRIMARY KEY(tenant, conversation, run))')

    def db(self):
        return sqlite3.connect(self.path, timeout=1)

    @contextmanager
    def locked(self):
        lock = sqlite3.connect(self.path + '.locks', timeout=1)
        try:
            lock.execute('BEGIN IMMEDIATE')
            yield
        finally:
            lock.rollback()
            lock.close()

    def identity(self, token):
        if not token:
            raise ContextError('missing_execution_context')
        scope = self.verify(token)
        if (not isinstance(scope, VerifiedExecution)
                or type(scope.tenant_id) is not int or scope.tenant_id <= 0
                or type(scope.conversation_id) is not int or scope.conversation_id <= 0
                or not scope.principal_id):
            raise ContextError('unverified_execution_context')
        return scope

    def load(self, scope, run_id):
        with self.db() as db:
            row = db.execute('SELECT state FROM bounded_runs WHERE '
                             'tenant=? AND conversation=? AND run=?',
                             (scope.tenant_id, scope.conversation_id, run_id)).fetchone()
        return json.loads(row[0]) if row else None

    def save(self, s):
        from anchor_ai.pii import redact_value
        s = redact_value(s)
        with self.db() as db:
            db.execute('INSERT OR REPLACE INTO bounded_runs VALUES(?,?,?,?)',
                       (s['tenant_id'], s['conversation_id'], s['run_id'], json.dumps(s)))

    def event(self, s, subsystem, **fields):
        event = dict(sequence=len(s['events']) + 1, subsystem=subsystem,
                     trace_id=s['trace_id'], tenant_id=s['tenant_id'],
                     conversation_id=s['conversation_id'], step=s['steps'], **fields)
        s['events'].append(event)
        from anchor_ai.pii import redact_value
        event = redact_value(event)
        s['events'][-1] = event
        self.emit(copy.deepcopy(event))

    def stop(self, s, status, reason):
        s['status'] = status
        s['escalation'] = dict(reason=reason, tenant_id=s['tenant_id'],
                              conversation_id=s['conversation_id'],
                              plan=copy.deepcopy(s['plan']), phase=s['phase'],
                              orchestration=copy.deepcopy(s.get('orchestration')),
                              caps=copy.deepcopy(s['caps']), cost_units=s['cost_units'],
                              memory=copy.deepcopy(s['memory']),
                              attempts=copy.deepcopy(s['attempts']),
                              proposed_action=s['arguments'], idempotency_key=s['key'],
                              tokens=s['tokens'], steps=s['steps'], calls=s['calls'])
        if status in TERMINAL:
            from anchor_ai.escalation import bounded_handoff
            s['escalation'].update(bounded_handoff(s, reason))
        self.event(s, 'escalation', reason=reason, persisted=True)
        active = CURRENT.get()
        ctx = None if active else {'trace_id': s['trace_id'], 'request_id': s['run_id'],
                                    'parent_span_id': None}
        with span('python.bounded_agent.handoff', ctx=ctx, tenant_id=s['tenant_id'],
                  conversation_id=s['conversation_id']):
            digest = argument_fingerprint(s['arguments'])
            self.event(s, 'trajectory', decision=trajectory_decision(
                kind='handoff', claim='human_handoff', human_required=True,
                argument_fingerprint=digest, expected_argument_fingerprint=digest,
                handoff_complete=all(k in s['escalation'] for k in
                    ('reason', 'plan', 'memory', 'attempts', 'caps', 'proposed_action',
                     'tenant_id', 'conversation_id', 'orchestration'))))
        self.save(s)
        return s

    def allowed(self, scope, action, arguments):
        tenant = self.policy['tenants'].get(scope.tenant_id, {})
        names = tenant.get('roles', {}).get(scope.role, [])
        return (action in names and action in self.policy['tools']
                and (action != 'issue_refund' or
                     arguments['amount_cents'] <= tenant.get('refund_ceiling_cents', 0)))

    def review(self, token, run_id, approved):
        scope = self.identity(token)
        if scope.role != 'refund_reviewer':
            raise ContextError('reviewer_required')
        with self.locked():
            s = self.load(scope, run_id)
            if not s:
                raise ContextError('run_not_in_scope')
            if s['status'] != 'awaiting_approval':
                return s
            self.event(s, 'review', approved=approved, reviewer=scope.principal_id)
            if not approved:
                return self.stop(s, 'rejected', 'approval_rejected')
            s['approval'] = dict(reviewer=scope.principal_id, digest=s['digest'])
            s['status'] = 'running'
            self.save(s)
            return s

    async def execute(self, token, request):
        scope = self.identity(token)
        req = AgentRequest.model_validate(request)
        if req.operation in ('approve', 'reject'):
            return self.review(token, req.run_id, req.operation == 'approve')
        from anchor_ai.jailbreak_guard import evaluate, REFUSAL
        if req.operation == 'start' and evaluate(req.message).refused:
            return dict(status='denied', reply=REFUSAL, calls=0)
        if req.operation == 'start':
            try:
                req.message = hardened_customer_text(req.message)
            except ValueError:
                return dict(status='denied', reply=REFUSAL, calls=0)
        with self.locked():
            s = self.load(scope, req.run_id)
            if s is None:
                if req.operation != 'start':
                    raise ContextError('run_not_in_scope')
                if not req.message.strip():
                    raise ValueError('message_required')
                strategy = self.policy['bounded']['strategies'][req.ticket_type]
                args = {'order_id': req.order_id, 'amount_cents': req.amount_cents}
                key = hashlib.sha256(json.dumps(
                    [scope.tenant_id, scope.conversation_id, req.run_id, args],
                    sort_keys=True).encode()).hexdigest()
                s = dict(run_id=req.run_id, tenant_id=scope.tenant_id,
                         conversation_id=scope.conversation_id,
                         trace_id=(CURRENT.get().record['trace_id'] if CURRENT.get() else key[:32]),
                         strategy=strategy, message=req.message, arguments=args,
                         key=key, digest=key, status='running', phase=0,
                         plan=ACTIONS[:] if req.ticket_type == 'multi-step refund' else ['retrieve'],
                         steps=0, calls=0, cost_units=0, tokens=0, memory=[],
                         attempts=[], events=[], pending=None, approval=None,
                         caps=dict(self.policy['run']), started=time.time())
                s['orchestration'] = select_flow(req.ticket_type)
                self.event(s, 'planner_selection', strategy=strategy)
                self.save(s)
            elif req.operation == 'start':
                if (req.message != s['message'] or req.order_id != s['arguments']['order_id']
                        or req.amount_cents != s['arguments']['amount_cents']):
                    raise ValueError('run_arguments_changed')
            if s['status'] in TERMINAL or s['status'] == 'awaiting_approval':
                return s
            if evaluate(s['message']).refused:
                s['reply'] = REFUSAL
                return self.stop(s, 'denied', 'jailbreak_detected')
            self.event(s, 'memory_read', records=len(s['memory']))
            self.save(s)
            # A resumed workflow retains its durable run identity, but this HTTP
            # execution belongs to the current caller's trace, not a synthetic root.
            ctx = None if CURRENT.get() else {
                'trace_id': s['trace_id'], 'request_id': s['run_id'], 'parent_span_id': None}
            with span('python.bounded_agent', ctx=ctx, stage='agent', tenant_id=scope.tenant_id,
                      conversation_id=scope.conversation_id):
                return await self.loop(scope, s)

    def incident_allowed(self, scope):
        if self.incident_check is None:
            return True
        try:
            return self.incident_check(scope.tenant_id) is True
        except Exception:
            return False

    async def loop(self, scope, s):
        # Also protect internal/delegated callers entering from a saved checkpoint.
        if (s['status'] in TERMINAL
                or s.get('escalation', {}).get('do_not_retry') is True):
            return s
        while s['phase'] < len(s['plan']):
            if not self.incident_allowed(scope):
                s['reply'] = ('Automatic processing is unavailable. Your ticket is being handed '
                              'to a human; we will verify any pending refund before further action.')
                return self.stop(s, 'denied', 'human_only')
            if time.time() - s['started'] >= s['caps']['deadline_seconds']:
                return self.stop(s, 'exhausted', 'deadline')
            if s['pending'] is None:
                reserve = self.model.reservation
                permitted = (s['steps'] < s['caps']['max_steps']
                             and s['tokens'] + reserve <= s['caps']['max_tokens'])
                self.event(s, 'budget_check', permitted=permitted,
                           steps=s['steps'], tokens=s['tokens'], reserve=reserve)
                if not permitted:
                    return self.stop(s, 'exhausted', 'step_or_token_budget')
                s['steps'] += 1
                s['tokens'] += reserve
                self.save(s)
                try:
                    worker = worker_for(s)
                    if worker is not None:
                        self.event(s, 'worker_assignment', worker=worker, action=s['plan'][s['phase']])
                    with span('python.bounded_agent.model', stage='llm'):
                        action, used = await asyncio.wait_for(model_call(propose, self.model, s, reserve),
                            timeout=self.policy['bounded']['call_timeout_seconds'])
                    if type(used) is not int or not 0 <= used <= reserve:
                        raise ValueError('invalid_usage')
                    s['tokens'] -= reserve - used
                    self.event(s, 'planner', action=action, charged=used)
                except Exception:
                    return self.stop(s, 'failed', 'model_error')
                allowed = self.allowed(scope, action, s['arguments'])
                allowed = allowed and action == s['plan'][s['phase']]
                self.event(s, 'guardrail', action=action, allowed=allowed)
                attempt = dict(action=action, allowed=allowed, result='proposed')
                s['attempts'].append(attempt)
                if not allowed:
                    self.event(s, 'memory_write', outcome='denied')
                    return self.stop(s, 'denied', 'unauthorized_action')
                s['pending'] = dict(action=action, reserved=False)
                self.save(s)
            action = s['pending']['action']
            if not self.allowed(scope, action, s['arguments']):
                return self.stop(s, 'denied', 'authorization_changed')
            if action == 'issue_refund':
                if not s['approval']:
                    return self.stop(s, 'awaiting_approval', 'approval_required')
                if s['approval']['digest'] != s['digest']:
                    return self.stop(s, 'denied', 'approval_scope_mismatch')
            from anchor_ai.harness import before_dispatch
            if not await before_dispatch(self, scope, s, action):
                return self.stop(s, 'denied', 'harness_policy_denied')
            if not s['pending']['reserved']:
                cost = self.policy['tools'][action]['cost_units']
                permitted = (s['calls'] < s['caps']['max_calls']
                             and s['cost_units'] + cost <= s['caps']['max_cost_units'])
                self.event(s, 'dispatch_budget', permitted=permitted, action=action)
                if not permitted:
                    return self.stop(s, 'exhausted', 'call_or_cost_budget')
                s['calls'] += 1
                s['cost_units'] += cost
                s['pending']['reserved'] = True
                self.save(s)
            if not self.incident_allowed(scope):
                return self.stop(s, 'denied', 'human_only')
            try:
                with span('python.bounded_agent.tool',
                          stage='retrieval' if action == 'retrieve' else 'tool', tool_name=action):
                    if action == 'retrieve':
                        observation = await asyncio.wait_for(self.retrieve(scope, s['message']),
                            timeout=self.policy['bounded']['call_timeout_seconds'])
                        if (observation['tenant_id'] != scope.tenant_id
                                or any(h.get('tenant_id', scope.tenant_id) != scope.tenant_id
                                       for h in observation['hits'])):
                            raise ValueError('tenant_memory_leak')
                    else:
                        observation = await asyncio.wait_for(self.backend.call(
                            scope, action, s['arguments'], s['key']),
                            timeout=self.policy['bounded']['call_timeout_seconds'])
                        if observation.get('order_id') != s['arguments']['order_id']:
                            raise ValueError('receipt_scope_mismatch')
                        if action == 'check_refund' and observation.get('eligible') is not True:
                            raise ValueError('refund_ineligible')
                        if action == 'issue_refund' and (
                                observation.get('status') != 'refunded'
                                or observation.get('idempotency_key') != s['key']
                                or observation.get('amount_cents') != s['arguments']['amount_cents']):
                            raise ValueError('invalid_refund_receipt')
                    observation = hardened_observation(observation)
                    digest = argument_fingerprint(s['arguments'])
                    self.event(s, 'trajectory', decision=trajectory_decision(
                        kind='tool', tool=action, expected_tool=s['plan'][s['phase']],
                        argument_fingerprint=digest, expected_argument_fingerprint=digest,
                        evidence=[m['action'] for m in s['memory']], claim='evidence_available',
                        confirmed=bool(s['approval'] and s['approval']['digest'] == s['digest']),
                        dry_run=False, reversible=False))
            except Exception:
                s['attempts'][-1]['result'] = 'error'
                self.event(s, 'memory_write', outcome='error')
                return self.stop(s, 'failed', 'tool_error')
            s['attempts'][-1]['result'] = 'success'
            self.event(s, 'tool_dispatch', action=action, observation=observation,
                       idempotency_key=s['key'] if action == 'issue_refund' else None)
            s['memory'].append(dict(action=action, observation=observation))
            self.event(s, 'memory_write', outcome='success')
            if action == 'retrieve' and not observation['hits']:
                return self.stop(s, 'exhausted', 'evidence_exhausted')
            s['phase'] += 1
            s['pending'] = None
            self.save(s)
        s['status'] = 'completed'
        self.event(s, 'complete', calls=s['calls'], tokens=s['tokens'])
        self.save(s)
        return s


def replay_trajectory_fixture(fixture, route, *, delegated=False):
    """Offline evidence adapter; no tools, authorization, or money movement."""
    events = []
    ctx = {'trace_id': hashlib.sha256(fixture['id'].encode()).hexdigest()[:32],
           'request_id': fixture['id'], 'parent_span_id': None}
    with span('python.fixture.bounded', ctx=ctx, tenant_id=fixture['tenant_id']):
        for raw in fixture['steps']:
            worker = span('python.fixture.worker') if delegated else nullcontext()
            with worker:
                with span('python.fixture.decision'):
                    tool = raw.get('tool', 'none')
                    if tool == 'refund_route':
                        tool = route
                    digest = argument_fingerprint(raw.get('arguments', fixture['arguments']))
                    expected = argument_fingerprint(fixture['arguments'])
                    decision = trajectory_decision(
                        kind=raw['kind'], tool=tool, expected_tool=raw.get('expected_tool', 'none'),
                        argument_fingerprint=digest, expected_argument_fingerprint=expected,
                        evidence=raw.get('evidence', []), claim=raw['claim'],
                        confirmed=raw.get('confirmed', False), dry_run=True, reversible=False,
                        human_required=raw.get('human_required', False),
                        handoff_complete=raw.get('handoff_complete', False), success=raw.get('success', True))
                    events.append(dict(sequence=len(events) + 1, subsystem='trajectory',
                                       tenant_id=fixture['tenant_id'], decision=decision))
    return dict(endpoint='bounded_agent', offline_fixture=True,
                tenant_id=fixture['tenant_id'], events=events, outcome_correct=fixture['outcome_correct'])


def voice_authority():
    """Voice is read-only; prose cannot grant a tool approval."""
    return {'confirmed': False, 'dispatch_allowed': False, 'approval_required': True}


class SemanticAnswerRuntime:
    """Shared read-only boundary; never invoke financial tools from generate.

    verify(token) returns server-verified identity with tenant_id and role.
    resolve(identity, query, path) authorizes and returns Snapshot.
    embed(query, snapshot) returns a vector in snapshot.embedding_version.
    await generate(identity, query, snapshot) returns Generated provenance.
    """
    def __init__(self, *, verify, resolve, embed, generate, cache,
                 fixture_adapters=True, production=False, release_evidence=None):
        required = {'authenticated_paths', 'embedding_calibration', 'ann_recall'}
        if production and (fixture_adapters or not release_evidence
                           or any(release_evidence.get(k) is not True for k in required)):
            raise ValueError('semantic_cache_release_evidence_missing')
        self.verify, self.resolve = verify, resolve
        self.embed, self.generate, self.cache = embed, generate, cache

    async def answer(self, token, query, *, path):
        from anchor_ai.semantic_cache import Snapshot, Generated
        if not token or path not in {'chat', 'rag'}:
            raise ContextError('cache_context_rejected')
        identity = self.verify(token)
        if (identity is None or type(identity.tenant_id) is not int
                or identity.tenant_id <= 0 or not identity.role):
            raise ContextError('cache_identity_rejected')
        from anchor_ai.jailbreak_guard import evaluate, REFUSAL
        if evaluate(query).refused:
            return REFUSAL
        def fresh():
            s = self.resolve(identity, query, path)
            if (not isinstance(s, Snapshot) or s.tenant_id != identity.tenant_id
                    or s.role != identity.role or s.answer_path != path):
                raise ContextError('cache_scope_rejected')
            return s
        from anchor_ai.injection_filter import screen_customer
        _, flags = screen_customer(query)
        if flags:
            return REFUSAL
        snapshot = fresh()
        eligible = self.cache.eligible(snapshot, query) == 'public_faq'
        vector = self.embed(query, snapshot) if eligible else None
        hit = self.cache.lookup(snapshot, query, vector)
        current = fresh()
        if hit is not None and current == snapshot:
            return hit
        if current != snapshot:
            snapshot = current
            eligible = self.cache.eligible(snapshot, query) == 'public_faq'
            vector = self.embed(query, snapshot) if eligible else None
        generated = await model_call(self.generate, identity, query, snapshot)
        if not isinstance(generated, Generated):
            raise ValueError('missing_answer_provenance')
        if eligible and fresh() == snapshot:
            self.cache.store(snapshot, query, vector, generated)
        return generated.text


def create_batch_runtime(*, adapter, verify, path, config_path='config/batching.yml',
                         emit=None, production=False, evidence=None, tokenizer=None):
    """Composition seam; authorization and production evidence stay mandatory."""
    from anchor_ai.batching import BatchRuntime, Scheduler, configuration
    scheduler = Scheduler(adapter, configuration(config_path), emit=emit)
    return BatchRuntime(scheduler, verify=verify, path=path, production=production,
                        evidence=evidence, tokenizer=tokenizer)


@contextmanager
def verified_cost_scope(collector, *, verify, token, feature, role, model,
                        logical_call_id=None, window=None):
    """verify is a trusted execution-token verifier, never request-supplied code."""
    if not token:
        raise ContextError('missing_execution_context')
    identity = verify(token)
    if (identity is None or type(identity.tenant_id) is not int
            or identity.tenant_id <= 0 or not identity.role):
        raise ContextError('unverified_execution_context')
    with collector.scope(tenant_id=identity.tenant_id, feature=feature, role=role,
                         model=model, logical_call_id=logical_call_id, window=window):
        yield identity


@contextmanager
def verified_quota_scope(*, verify, token, client_for, feature, bound):
    """client_for binds verified identity to the Rails authority context."""
    from anchor_ai.quota_client import quota_scope
    if not token:
        raise ContextError('missing_execution_context')
    identity = verify(token)
    if (identity is None or type(identity.tenant_id) is not int
            or identity.tenant_id <= 0 or not identity.role):
        raise ContextError('unverified_execution_context')
    with quota_scope(client_for(identity, feature), bound=bound):
        yield identity


def initialize_provider_resilience(router, *, database, **options):
    """Call once per worker; this grants no identity or admission allowance."""
    from anchor_ai.observability import emit_provider_resilience
    if hasattr(router, 'resilience'):
        raise ValueError('provider_resilience_already_initialized')
    options.setdefault('emit', emit_provider_resilience)
    return router.enable_resilience(database=database, **options)


# Incident composition: deterministic offline sends, existing product retry loop.
INCIDENT_POLICY = Path('labs/failover-budget-incident/broken_policy.json')
INCIDENT_DATASET = Path('eval/datasets/failover_budget_incident.jsonl')


def corrected_incident_policy():
    from anchor_ai.provider import routing_configuration
    return copy.deepcopy(routing_configuration()['incident_policy'])


class IncidentClock:
    def __init__(self):
        self.now = 0.0
    def __call__(self):
        return self.now
    async def sleep(self, delay):
        self.now += delay


class IncidentRouter:
    """Single-send fixture; never selects a live adapter."""
    def __init__(self, authority, clock, order):
        self.authority, self.clock = authority, clock
        self.config = dict(order=order, logical_model='incident-fixture', deployments={
            name: dict(model='incident-fixture', region='eu') for name in order})
        self.calls, self.served_by = [], None

    def eligible(self, name, tenant_id, payload):
        return tenant_id == self.authority.tenant and name in self.config['order']

    async def physical(self, name, request, payload):
        import httpx
        import uuid
        from anchor_ai.provider import ProviderError
        ticket, context = payload['ticket'], payload['context']
        reservation = self.authority.call('reserve', ticket=ticket, context=context,
                                          attempt_id=uuid.uuid4().hex, bound=1)
        if not reservation['allowed']:
            raise ProviderError(reservation['reason'], status=503)
        self.clock.now += .02
        self.calls.append(dict(ticket=ticket, provider=name, attempt=len(self.calls) + 1,
            window=reservation['window'], used_before=reservation['used_before'],
            input_tokens=128, active_concurrency=1,
            deadline_remaining=round(.5 - self.clock.now, 4)))
        # Admission and this synthetic send have no intervening await.
        # This does not establish race-free distributed live dispatch.
        return httpx.Response(503 if name == 'primary' else 200,
                              json={'usage': {'prompt_tokens': 128}})


class IncidentRuntime:
    def __init__(self, directory, policy=None, *, production=False):
        from anchor_ai.provider import validate_incident_policy
        from anchor_ai.quota_client import IncidentAuthority
        from anchor_ai.resilience import CircuitStore, ResilienceBoundary, configuration
        if production:
            raise ValueError('incident_release_evidence_missing')
        policy = corrected_incident_policy() if policy is None else policy
        validate_incident_policy(policy)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.authority = IncidentAuthority(self.directory / 'authority.sqlite3')
        self.clock = IncidentClock()
        self.router = IncidentRouter(self.authority, self.clock, policy['order'])
        resilience = configuration()
        resilience['max_attempts'] = policy['max_attempts']
        resilience['total_seconds'] = policy['total_seconds']
        self.events = []
        self.boundary = ResilienceBoundary(self.router,
            CircuitStore(self.directory / 'circuits.sqlite3', resilience, clock=self.clock),
            resilience, clock=self.clock, sleep=self.clock.sleep,
            jitter=lambda: 0, emit=self.events.append)

    async def ticket(self, ticket, context):
        import httpx
        from anchor_ai.provider import ProviderError
        self.clock.now = 0
        self.authority.call('queue', ticket=ticket, context=context)
        request = httpx.Request('POST', 'https://offline.invalid/incident',
                               json=dict(ticket=ticket, context=context))
        try:
            response = await self.boundary.route(request, self.authority.tenant)
            return dict(ticket=ticket, status=response.status_code)
        except (ProviderError, TimeoutError):
            self.authority.call('handoff', ticket=ticket, context=context)
            return dict(ticket=ticket, status='human')


def broken_trace():
    pending, rows, clock, cumulative = [], [], 0.0, 0
    for round_number in range(6):
        pending.append(dict(ticket='A17-' + str(round_number), arrival=round_number))
        clock = max(clock, round_number)
        latencies = []
        for ticket in pending:
            clock += .5
            cumulative += 1
            latencies.append(clock - ticket['arrival'])
            rows.append(dict(event='attempt', ticket=ticket['ticket'], provider='primary',
                reset='per_request', used_before=0, window=0, breaker='closed',
                input_tokens=128, active_concurrency=1, deadline_remaining=None))
        rows.append(dict(event='series', round=round_number + 1, spend_units=len(pending),
            cumulative_units=cumulative, max_attempt_latency_s=max(latencies), pending=len(pending)))
    assert cumulative == 21
    return rows


async def recovered_trace(directory, human=False):
    runtime = IncidentRuntime(directory)
    context = dict(message='Where is order A17?', evidence=['order:A17'],
                   history=['customer asked for delivery status'], proposed_action='lookup_order')
    series = []
    if human:
        runtime.authority.call('queue', ticket='queued-A17', context=context)
        runtime.authority.call('human_only')
    for index in range(6):
        before = len(runtime.router.calls)
        result = await runtime.ticket('A17-' + str(index), copy.deepcopy(context))
        snapshot = runtime.authority.call('snapshot')
        series.append(dict(event='series', round=index + 1,
            spend_units=len(runtime.router.calls) - before, cumulative_units=snapshot['used'],
            max_attempt_latency_s=round(runtime.clock.now, 4), status=result['status']))
    snapshot = runtime.authority.call('snapshot')
    assert snapshot['used'] <= 12
    if human:
        assert not runtime.router.calls
        assert all(t['state'] == 'human' and t['context'] == context for t in snapshot['tickets'])
    for call in runtime.router.calls:
        call.update(event='attempt')
    return dict(calls=runtime.router.calls, decisions=runtime.events, series=series,
                authority=snapshot,
                limitation='Offline one-host admission only; hosted dispatch and multi-host coordination not verified')


def incident_regression_cases():
    from anchor_ai.provider import incident_policy_violations
    cases = [json.loads(line) for line in INCIDENT_DATASET.read_text().splitlines() if line.strip()]
    assert {c['id'] for c in cases} == {'window', 'cycle', 'owners', 'human'}
    for case in cases:
        if case['id'] == 'human':
            assert case['scenario'] == 'human_only' and case['expected_provider_calls'] == 0
            continue
        policy = corrected_incident_policy()
        policy.update(case['mutation'])
        assert case['expected_error'] in incident_policy_violations(policy)
    return len(cases)


def incident_main():
    import argparse
    import tempfile
    from anchor_ai.provider import incident_policy_violations
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['broken', 'validate', 'corrected', 'human', 'all'])
    mode = parser.parse_args().mode
    if mode in ('broken', 'all'):
        for row in broken_trace():
            print(json.dumps(row))
    if mode in ('validate', 'all'):
        errors = incident_policy_violations(json.loads(INCIDENT_POLICY.read_text()))
        assert errors == ['cyclic_routing', 'conflicting_retry_ownership', 'quota_window_misconfiguration']
        print(json.dumps(dict(event='planted_policy_rejected', errors=errors,
                              regression_cases=incident_regression_cases())))
    for scenario in ('corrected', 'human'):
        if mode in (scenario, 'all'):
            with tempfile.TemporaryDirectory() as directory:
                report = asyncio.run(recovered_trace(directory, human=scenario == 'human'))
                print(json.dumps(dict(event=scenario, **report)))
    print('Offline smoke only: provider network, hosted quality, and multi-host emergency dispatch not verified; production release remains blocked.')
