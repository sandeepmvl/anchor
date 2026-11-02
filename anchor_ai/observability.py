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
    item = Span(name, ctx, **fields)
    token = CURRENT.set(item)
    try:
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
