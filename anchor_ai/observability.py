import contextvars
import json
import logging
import re
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal

CURRENT = contextvars.ContextVar('anchor_span', default=None)
LOGGER = logging.getLogger('anchor.telemetry')
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(message)s'))
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False
SINK = lambda record: LOGGER.info(json.dumps(record, separators=(',', ':')))
# USD / million tokens. Verify against https://openai.com/api/pricing/ before live use.
PRICE_VERSION = 'gpt-4o-mini-standard-2024-07'
PRICES = {'gpt-4o-mini': {'input': '0.15', 'cached_input': '0.075',
                         'cache_write_input': '0', 'output': '0.60'},
          'anchor-reasoning-replay': {'input': '1.25', 'cached_input': '0',
                                     'cache_write_input': '0', 'output': '10'}}
FIELDS = {'tenant_id', 'conversation_id', 'stage', 'status_code', 'request_model',
          'response_model', 'provider_request_id', 'tokens', 'cost_estimate',
          'currency', 'cost_basis', 'price_version', 'attempts',
          'reasoning_setting', 'reasoning_effort', 'max_completion_tokens',
          'finish_reason', 'completion_status', 'fixture_latency_ms',
          'fixture_latency_basis', 'ticket_id', 'ticket_class',
          'prompt_version', 'prompt_fingerprint', 'temperature', 'max_tokens',
          'counter', 'context_margin', 'section_tokens', 'section_caps',
          'section_allocations', 'section_priorities', 'estimated_input_tokens',
          'output_reserve', 'context_window', 'pre_history_tokens', 'summary_calls',
          'summary_input_tokens', 'summary_output_cap', 'summary_version',
          'folded_position', 'pending_messages', 'context_fallback'}
FIELDS.update({'server_ttft_ms', 'stream_total_ms', 'pacing_peak_chars'})
# These are producer-local durations, not client delivery measurements.
ERRORS = {'validation_failed', 'service_auth_failed', 'service_not_configured',
          'internal_error', 'provider_bad_request', 'provider_auth_failed',
          'provider_forbidden', 'provider_not_found', 'provider_quota_exhausted',
          'provider_rate_limited', 'provider_unknown_429', 'provider_unavailable',
          'provider_transport_error', 'provider_timeout', 'invalid_provider_response',
          'missing_provider_key', 'invalid_provider_mode', 'provider_rejected',
          'invalid_reasoning_effort', 'invalid_reasoning_limit',
          'unsupported_reasoning_model', 'replay_miss', 'client_disconnect',
          'context_budget_exceeded'}

# Diagnostic metadata only: never include scripts, environment, or stdout.
FIELDS.update({'sandbox_backend', 'sandbox_exit_reason', 'sandbox_output_bytes'})
ERRORS.update({'isolation_unavailable', 'capacity_unavailable', 'tenant_quota',
               'unapproved_script', 'sandbox_unauthorized', 'confirmation_required',
               'egress_denied', 'invalid_tenant', 'invalid_deadline'})


def now():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def safe_id(value):
    return value if isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,80}', value) else None


def safe_model_id(value):
    return value if isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_.-]{1,80}', value) else None


def context(headers):
    value = headers.get('traceparent', '')
    match = re.fullmatch(r'00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})', value)
    valid = match and int(match[1], 16) and int(match[2], 16)
    return {'request_id': safe_id(headers.get('x-request-id')) or uuid.uuid4().hex,
            'trace_id': match[1] if valid else uuid.uuid4().hex,
            'parent_span_id': match[2] if valid else None}


def tokens_from_usage(usage):
    if usage is None:
        return None
    data = usage.model_dump() if hasattr(usage, 'model_dump') else usage
    prompt, output = data['prompt_tokens'], data['completion_tokens']
    cached = (data.get('prompt_tokens_details') or {}).get('cached_tokens', 0)
    reasoning = (data.get('prompt_tokens_details') or {}).get('unused_reasoning')
    reasoning = (data.get('completion_tokens_details') or {}).get('reasoning_tokens')
    if cached > prompt or (reasoning is not None and reasoning > output):
        raise ValueError('invalid usage subsets')
    return {'input': prompt - cached, 'cached_input': cached, 'cache_write_input': 0,
            'output': output, 'reasoning': reasoning}


def estimate(model, tokens, prices=None):
    if tokens is None:
        return None
    rates = (PRICES if prices is None else prices).get(model)
    if rates is None:
        return None
    return float(sum(Decimal(tokens[k]) * Decimal(rates[k]) for k in
                     ('input', 'cached_input', 'cache_write_input', 'output')) / Decimal(1000000))


class Span:
    def __init__(self, name, ctx=None, **fields):
        parent = CURRENT.get()
        ctx = ctx or ({'request_id': parent.record['request_id'],
                       'trace_id': parent.record['trace_id'],
                       'parent_span_id': parent.record['span_id']} if parent else context({}))
        self.record = dict(event='span', name=name, **ctx, span_id=uuid.uuid4().hex[:16],
                           started_at=now(), status='ok', error_type=None,
                           error_message=None, tokens=None, cost_estimate=None)
        self.started = time.monotonic()
        if parent:
            for key in ('tenant_id', 'conversation_id'):
                if key in parent.record:
                    fields.setdefault(key, parent.record[key])
        self.set(**fields)

    def set(self, **fields):
        for key, value in fields.items():
            if key not in FIELDS:
                raise ValueError('non-allowlisted telemetry field')
            if key in {'request_model', 'response_model'}:
                value = safe_model_id(value)
            if key in {'provider_request_id', 'ticket_id', 'ticket_class', 'task_id', 'call_id'}:
                value = safe_id(value)
            if key == 'prompt_version' and not (isinstance(value, str) and re.fullmatch(r'chat:v[1-9][0-9]*', value)):
                raise ValueError('invalid prompt version')
            if key == 'prompt_fingerprint' and not (isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value)):
                raise ValueError('invalid prompt fingerprint')
            if key in {'tenant_id', 'conversation_id'} and (type(value) is not int or value <= 0):
                continue
            if key == 'response_model' and value == 'anchor-reasoning-replay':
                self.record.update(price_version='fixture-illustrative-v1',
                                   cost_basis='illustrative_fixture_estimate')
            self.record[key] = value

    def fail(self, code, status=None):
        code = code if code in ERRORS else 'internal_error'
        self.record.update(status='error', error_type=code, error_message=code)
        if status is not None:
            self.set(status_code=status)

    def finish(self):
        self.record.update(ended_at=now(), latency_ms=round((time.monotonic() - self.started) * 1000, 3))
        SINK(dict(self.record))


@contextmanager
def span(name, ctx=None, **fields):
    # Summary events emitted after work must not impersonate retrieval timers.
    timing_name = {'tool': 'tool', 'llm': 'model', 'retrieval': 'retrieval'}.get(
        fields.get('stage'), 'application')
    if name == 'python.dense_rag':
        timing_name = 'application'
    item = Span(name, ctx, **fields)
    token = CURRENT.set(item)
    try:
        with segment(timing_name):
            yield item
    except BaseException as error:
        item.fail(getattr(error, 'code', 'internal_error'), getattr(error, 'status', None))
        raise
    finally:
        CURRENT.reset(token)
        item.finish()


RETRIEVAL_FIELDS = {'generation_id', 'embedding_model', 'embedding_revision',
                    'metric', 'k', 'overfetch', 'post_filter_count', 'filter_path',
                    'chunk_ids', 'scores', 'embedding_ms', 'search_ms', 'fallback_ms',
                    'evidence_chunk_ids', 'evidence_bytes', 'citations', 'abstained',
                    'generation_ms'}
FIELDS.update(RETRIEVAL_FIELDS)
ERRORS.update({'ingestion_failed', 'retrieval_not_configured', 'retrieval_failed'})


def emit_retrieval(record):
    with span('python.dense_rag', stage='retrieval') as item:
        item.set(**{key: record[key] for key in RETRIEVAL_FIELDS if key in record})


FIELDS.update({'dense_ms', 'sparse_ms', 'fusion_ms', 'total_ms'})
ERRORS.add('retrieval_deadline')
FIELDS.update({'analysis_version', 'analysis_calls', 'subquery_count',
               'analysis_fallback', 'analysis_reason'})
FIELDS.update({'pair_count', 'rerank_ms'})

FIELDS.update({'agent_iteration', 'tool_calls', 'iterations_remaining',
               'tool_calls_remaining', 'agent_exit', 'decision_kind',
               'decision_summary', 'tool_name', 'agent_mode'})
ERRORS.update({'agent_invalid_decision', 'agent_scope_mismatch', 'agent_tool_failed'})
AGENT_SUMMARIES = {
    'retrieve': 'Request another targeted search.',
    'final': 'Policy returned a final answer without a tool call.',
    'sufficient': 'Policy declared the accumulated evidence sufficient.',
}

FIELDS.update({'query_fingerprint', 'duplicate_query', 'new_chunk_ids',
               'seen_chunk_ids', 'hops_remaining'})

FIELDS.add('tool_call_fingerprint')
TOOL_ERRORS = {'unknown_tool', 'invalid_json', 'invalid_arguments', 'execution_error'}
ERRORS.update(TOOL_ERRORS)


@contextmanager
def tool_span(scope, call):
    import hashlib
    name = call.name if call.name in {'lookup_order', 'check_refund'} else 'unknown'
    fingerprint = hashlib.sha256(call.call_id.encode()).hexdigest()
    with span('python.tool', stage='tool', tenant_id=scope.tenant_id,
              conversation_id=scope.conversation_id, tool_name=name,
              tool_call_fingerprint=fingerprint) as item:
        yield item


REACT_EVENT_FIELDS = {
    'agent_iteration', 'decision_summary', 'tool_name', 'query_fingerprint',
    'tool_calls', 'evidence_chunk_ids', 'agent_exit', 'observation_redacted',
}
FIELDS.update(REACT_EVENT_FIELDS)


def emit_react(event, emit, **metadata):
    """Export safe metadata, never model transcripts or raw tool results."""
    if event not in {'decision_summary', 'action', 'observation', 'exit'}:
        raise ValueError('invalid_react_event')
    if set(metadata) - REACT_EVENT_FIELDS:
        raise ValueError('unsafe_react_metadata')
    if 'decision_summary' in metadata and metadata['decision_summary'] not in AGENT_SUMMARIES.values():
        raise ValueError('unsafe_react_summary')
    with span('python.react.' + event, stage='agent', **metadata) as item:
        record = {'event': event, **metadata}
        for key in ('request_id', 'trace_id', 'span_id', 'parent_span_id'):
            record[key] = item.record[key]
        emit(record)


FIELDS.add('mcp_server_id')
ERRORS.add('mcp_rejected')


@contextmanager
def mcp_span(scope, server_id, call_id):
    import hashlib
    identity = safe_id(server_id)
    if identity is None:
        raise ValueError('invalid configured server identity')
    with span('python.mcp', stage='tool', tenant_id=scope.tenant_id,
              conversation_id=scope.conversation_id, mcp_server_id=identity,
              tool_call_fingerprint=hashlib.sha256(call_id.encode()).hexdigest()) as item:
        yield item


FIELDS.update({'execution_mode', 'task_id', 'call_id', 'interop_protocol',
               'permission_decision', 'permission_reason'})
ERRORS.add('delegation_rejected')
ERRORS.add('orchestration_not_configured')

TRAJECTORY_TOOLS = {'none', 'retrieve', 'lookup_order', 'check_refund', 'issue_refund'}
TRAJECTORY_CLAIMS = {'intake', 'evidence_available', 'observed_answer', 'human_handoff'}


def trajectory_decision(*, kind, tool='none', expected_tool='none',
                        argument_fingerprint, expected_argument_fingerprint,
                        evidence=(), claim, confirmed=False, dry_run=False,
                        reversible=False, human_required=False,
                        handoff_complete=False, success=True):
    """Safe projection only; authority and expected values come from trusted state."""
    current = CURRENT.get()
    if current is None or type(current.record.get('tenant_id')) is not int:
        raise ValueError('trajectory_scope_missing')
    if (kind not in {'intake', 'tool', 'answer', 'handoff'}
            or tool not in TRAJECTORY_TOOLS or expected_tool not in TRAJECTORY_TOOLS
            or claim not in TRAJECTORY_CLAIMS
            or not isinstance(evidence, (list, tuple))
            or any(x not in TRAJECTORY_TOOLS - {'none'} for x in evidence)):
        raise ValueError('unsafe_decision_evidence')
    for value in (argument_fingerprint, expected_argument_fingerprint):
        if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
            raise ValueError('unsafe_argument_fingerprint')
    flags = dict(confirmed=confirmed, dry_run=dry_run, reversible=reversible,
                 human_required=human_required, handoff_complete=handoff_complete,
                 success=success)
    if any(type(value) is not bool for value in flags.values()):
        raise ValueError('unsafe_decision_flags')
    result = dict(kind=kind, tool=tool, expected_tool=expected_tool,
                  argument_fingerprint=argument_fingerprint,
                  expected_argument_fingerprint=expected_argument_fingerprint,
                  evidence=list(evidence), claim=claim, **flags)
    for key in ('tenant_id', 'trace_id', 'span_id', 'parent_span_id'):
        result[key] = current.record.get(key)
    return result


FIELDS.update({'vision_backend', 'vision_usage_basis'})
ERRORS.update({'live_vision_not_verified', 'vision_context_rejected',
               'vision_request_rejected', 'invalid_vision_output', 'vision_provider_rejected'})


@contextmanager
def vision_span():
    with span('python.vision', stage='vision', vision_backend='offline',
              vision_usage_basis='not_measured_offline') as item:
        yield item


def realtime_event(session, scope, event, epoch, queue_depth, **fields):
    allowed = {'send_ms', 'expected', 'received'}
    events = {'send_done', 'queued', 'barge_in', 'stt_start', 'stt_done',
              'generation_start', 'answer_text', 'tts_start', 'tts_done', 'audio',
              'frame_gap', 'speech_start', 'end_of_speech', 'session_failure', 'session_closed'}
    if event not in events or set(fields) - allowed:
        raise ValueError('unsafe_realtime_metadata')
    if any(type(value) not in (int, float) for value in fields.values()):
        raise ValueError('unsafe_realtime_metadata')
    return dict(event=event, t=time.monotonic(), trace_id=session,
                tenant_id=scope['tenant_id'], ticket_id=scope['ticket_id'],
                epoch=epoch, queue_depth=queue_depth, **fields)


# Bounded local-clock latency trees. Never retain prompts or tool arguments.
import math
import threading
from pathlib import Path
import yaml

ACTIVE = contextvars.ContextVar('anchor_latency_profile', default=None)
PARENT = contextvars.ContextVar('anchor_latency_parent', default=None)
NAMES = {'request', 'application', 'retrieval', 'branch', 'embedding', 'search',
         'rerank', 'model', 'tool', 'transport', 'retry_wait'}
MAX_SPANS = 256
MAX_TEXT_EVENTS = 128


def union(intervals):
    end = None
    total = 0.0
    for start, stop in sorted(intervals):
        if stop < start:
            raise ValueError('negative_interval')
        if end is None or start > end:
            total += stop - start
        elif stop > end:
            total += stop - end
        end = stop if end is None else max(end, stop)
    return total


class Profile:
    def __init__(self, trace_id=None, clock=time.perf_counter):
        self.trace_id = trace_id or uuid.uuid4().hex
        if (not isinstance(self.trace_id, str) or len(self.trace_id) != 32
                or any(c not in '0123456789abcdef' for c in self.trace_id)
                or int(self.trace_id, 16) == 0):
            raise ValueError('invalid_trace_id')
        self.clock = clock
        self.origin = clock()
        self.nodes = []
        self.lock = threading.Lock()
        self.closed = False
        self.truncated = False

    @contextmanager
    def activate(self):
        active = ACTIVE.set(self)
        parent = PARENT.set(None)
        try:
            yield self
        finally:
            PARENT.reset(parent)
            ACTIVE.reset(active)

    @contextmanager
    def span(self, name):
        if name not in NAMES:
            raise ValueError('unsafe_span_name')
        with self.lock:
            retained = not self.closed and len(self.nodes) < MAX_SPANS
            if not retained:
                self.truncated = True
            node = {'id': len(self.nodes) if retained else None,
                    'parent': PARENT.get(), 'name': name,
                    'start': self.clock() - self.origin, 'end': None,
                    'round_trips': 0, 'text_events': [], 'cached_tokens': None,
                    'injected_inference_ms': None, 'status': 'ok',
                    'text_events_truncated': False}
            if retained:
                self.nodes.append(node)
        # A dropped span's retained descendants remain attached to its parent.
        parent = PARENT.set(node['id'] if retained else PARENT.get())
        try:
            yield node
        except BaseException:
            with self.lock:
                node['status'] = 'error'
            raise
        finally:
            with self.lock:
                node['end'] = self.clock() - self.origin
                if retained and node['parent'] is None:
                    self.closed = True
            PARENT.reset(parent)

    def text(self, node):
        """Timestamp observed text events; never store their content."""
        with self.lock:
            if self.closed or node['id'] is None:
                return
            if len(node['text_events']) < MAX_TEXT_EVENTS:
                node['text_events'].append(self.clock() - self.origin)
            else:
                node['text_events_truncated'] = True
                self.truncated = True

    def report(self):
        # Snapshot under a short lock. Never join threads or wait for work here.
        with self.lock:
            nodes = [dict(n, text_events=list(n['text_events'])) for n in self.nodes]
            truncated = self.truncated
        roots = [n for n in nodes if n['parent'] is None]
        if len(roots) != 1:
            raise ValueError('one_request_root_required')
        if roots[0]['end'] is None:
            raise ValueError('unfinished_span')
        by_id = {n['id']: n for n in nodes}
        bounds = {}
        incomplete = False
        for node in nodes:
            parent = bounds.get(node['parent'])
            left = node['start']
            right = node['end']
            if parent is not None:
                left = min(max(left, parent[0]), parent[1])
                right = parent[1] if right is None else min(right, parent[1])
            elif node['parent'] is not None:
                raise ValueError('invalid_parent')
            if right is None or right < left:
                raise ValueError('negative_interval')
            clipped = node['end'] is None or (left, right) != (node['start'], node['end'])
            incomplete = incomplete or clipped
            bounds[node['id']] = (left, right, clipped)
        children = {n['id']: [] for n in nodes}
        for node in nodes:
            if node['parent'] in by_id:
                children[node['parent']].append(bounds[node['id']][:2])
        rows = []
        for node in nodes:
            left, right, clipped = bounds[node['id']]
            inclusive = right - left
            exclusive = inclusive - union(children[node['id']])
            events = [t for t in node['text_events'] if left <= t <= right]
            rows.append({
                'trace_id': self.trace_id, 'id': node['id'],
                'parent': node['parent'], 'name': node['name'],
                'start_ms': left * 1000, 'inclusive_ms': inclusive * 1000,
                'exclusive_ms': max(0.0, exclusive) * 1000,
                'status': node['status'], 'interval_clipped': clipped,
                'round_trips': node['round_trips'],
                'ttft_ms': (events[0] - node['start']) * 1000 if events else None,
                'inter_token_span_ms': (events[-1] - events[0]) * 1000 if events else None,
                'text_event_gaps_ms': [(b - a) * 1000 for a, b in zip(events, events[1:])],
                'text_events_truncated': node['text_events_truncated'],
                'cached_tokens': node['cached_tokens'],
                'injected_inference_ms': node['injected_inference_ms'],
            })
        root = next(r for r in rows if r['parent'] is None)
        return {'trace_id': self.trace_id, 'total_ms': root['inclusive_ms'],
                'application_remainder_ms': root['exclusive_ms'],
                'round_trips': sum(r['round_trips'] for r in rows),
                'incomplete': incomplete, 'truncated': truncated, 'spans': rows}


@contextmanager
def segment(name):
    profile = ACTIVE.get()
    if profile is None:
        yield None
    else:
        with profile.span(name) as node:
            yield node


async def model_call(call, *args, **kwargs):
    """Count logical client invocations, not hidden provider retry attempts."""
    with segment('model') as node:
        if node is not None:
            node['round_trips'] = 1
        return await call(*args, **kwargs)


def distribution(values):
    if not values:
        return None
    ordered = sorted(values)
    return {name: ordered[math.ceil(len(ordered) * p) - 1]
            for name, p in [('p50', .50), ('p95', .95), ('p99', .99)]}


def configuration(path='config/performance.yml'):
    config = yaml.safe_load(Path(path).read_text())['profiling']
    if (type(config.get('enabled')) is not bool
            or config.get('record_content') is not False
            or config.get('percentile') != 'nearest_rank'
            or type(config.get('samples')) is not int
            or config['samples'] != 20
            or not 0 < config.get('remainder_limit', 0) <= .10):
        raise ValueError('unsafe_performance_configuration')
    return config


class LatencyBoundary:
    """ASGI boundary; elapsed server time is not customer delivery time."""
    def __init__(self, app, enabled=None, sink=None):
        self.application = app
        self.enabled = enabled
        self.sink = sink

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.application(scope, receive, send)
        enabled = configuration()['enabled'] if self.enabled is None else self.enabled
        if not enabled:
            return await self.application(scope, receive, send)
        headers = {k.decode('latin1'): v.decode('latin1')
                   for k, v in scope.get('headers', [])}
        ctx = context(headers)
        if ctx['parent_span_id'] is None:
            headers['traceparent'] = '00-' + ctx['trace_id'] + '-' + uuid.uuid4().hex[:16] + '-01'
        scope = dict(scope, headers=[(k.encode('latin1'), v.encode('latin1'))
                                    for k, v in headers.items()])
        profile = Profile(ctx['trace_id'])
        finished = False

        async def timed_send(message):
            nonlocal finished
            await send(message)
            if message['type'] == 'http.response.body' and not message.get('more_body', False):
                finished = True

        try:
            with profile.activate():
                with profile.span('request'):
                    await self.application(scope, receive, timed_send)
        finally:
            report = profile.report()
            report['response_completed'] = finished
            report['scope'] = 'python_asgi_not_ingress_or_client_delivery'
            # Keep the legacy SINK stream compatible with span consumers.
            # Full waterfalls go to the dedicated sink or structured logger.
            record = {'event': 'latency_profile', 'name': 'python.latency_profile', **report}
            if self.sink is not None:
                self.sink(record)
            else:
                LOGGER.info(json.dumps(record, separators=(',', ':')))
