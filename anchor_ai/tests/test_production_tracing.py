import asyncio
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from anchor_ai import observability as obs
from anchor_ai import runtime as runtime_module
from anchor_ai.contracts import AgentRequest, VerifiedExecution
from anchor_ai.hybrid_retriever import HybridRuntime, RecordedHybridClient
from anchor_ai.reranking import Reranker, OfflinePairScorer, DelayedTestScorer
from anchor_ai.runtime import BoundedRuntime


def rails_headers():
    # Actual Rails service files, with only external boundary dependencies stubbed.
    source = r'''
require 'active_support'
require 'active_support/core_ext/hash'
require 'json'
require 'ostruct'
require_relative 'app/services/ai_telemetry'
module Rails
  def self.env; OpenStruct.new(production?: false); end
end
class PiiRedactionClient
  def self.value(value); value; end
end
require_relative 'app/services/agent_orchestration_client'
records = []
AiTelemetry.sink = ->(record) { records << record }
headers = nil
transport = ->(_uri, request) {
  headers = {'traceparent' => request['traceparent'], 'x-request-id' => request['X-Request-Id']}
  OpenStruct.new(code: '200', body: '{}')
}
AiTelemetry.span('rails.request') do |root|
  AgentOrchestrationClient.new(secret: 'fixture-secret', transport: transport).execute(
    {run_id: 'observed-fixture'}, execution_context: 'trusted-fixture',
    request_id: 'fixture', traceparent: root.headers['traceparent'])
end
puts JSON.generate(headers: headers, records: records)
'''
    result = subprocess.run(['ruby', '-e', source], text=True, capture_output=True,
                            check=True, timeout=5)
    return json.loads(result.stdout)


def test_bounded_request_preserves_rails_trace_and_localizes_rerank(tmp_path, monkeypatch):
    monkeypatch.delenv('ANCHOR_TELEMETRY_ENDPOINT', raising=False)
    ruby = rails_headers()
    exported, legacy = [], []
    monkeypatch.setattr(obs, 'EXPORT_SINK', exported.append)
    monkeypatch.setattr(obs, 'SINK', legacy.append)
    client = RecordedHybridClient()
    hybrid = HybridRuntime(1, client.acquire_snapshot, client.dense_search,
        client.sparse_search, reranker=Reranker(DelayedTestScorer(
            OfflinePairScorer(), test_only=True)))
    policy = yaml.safe_load(Path('config/agent_policy.yml').read_text())
    ticket_type = next(key for key in policy['bounded']['strategies']
                       if key != 'multi-step refund')
    role = next(role for role, actions in policy['tenants'][1]['roles'].items()
                if 'retrieve' in actions)
    identity = VerifiedExecution(tenant_id=1, conversation_id=7,
        principal_id='fixture-principal', role=role)
    async def retrieve(scope, message):
        return await hybrid.retrieve(message)
    async def propose_fixture(model, state, reserve):
        return 'retrieve', 1
    async def fixture_dispatch(*args):
        return True
    # These trusted fixtures replace model decisions and harness dependencies,
    # not the runtime's budgets, scope checks, checkpointing, or retrieval work.
    monkeypatch.setattr(runtime_module, 'propose', propose_fixture)
    import anchor_ai.harness
    monkeypatch.setattr(anchor_ai.harness, 'before_dispatch', fixture_dispatch)
    runtime = BoundedRuntime(tmp_path / 'runs.sqlite3', verify=lambda token: identity,
        model=SimpleNamespace(reservation=1), retrieve=retrieve, backend=None)
    private = 'PRIVATE_PROMPT_SENTINEL alex@example.test'
    request = dict(operation='start', run_id='trace-fixture',
        ticket_type=ticket_type, message=private, order_id='A100', amount_cents=2500)
    async def execute():
        with obs.span('http.fixture', ctx=obs.context(ruby['headers'])):
            return await runtime.execute('trusted-execution-fixture', request)
    result = asyncio.run(execute())
    assert result['status'] == 'completed', result['status']
    agent = next(r for r in legacy if r['name'] == 'python.bounded_agent')
    http = next(r for r in legacy if r['name'] == 'http.fixture')
    rails = next(r for r in ruby['records'] if r.get('name') == 'rails.agent_orchestration')
    assert agent['trace_id'] == http['trace_id'] == rails['trace_id']
    assert http['parent_span_id'] == rails['span_id']
    assert agent['parent_span_id'] == http['span_id']
    by_id = {r['span_id']: r for r in legacy}
    for record in legacy:
        if record['parent_span_id'] in by_id:
            assert by_id[record['parent_span_id']]['trace_id'] == record['trace_id']
    rerank = next(r for r in exported if r['name'] == 'python.reranker')
    assert rerank['latency_ms'] >= 150
    parent_ids = {r['parent_span_id'] for r in exported}
    leaves = [r for r in exported if r['span_id'] not in parent_ids]
    slowest = max(leaves, key=lambda r: r['latency_ms'])
    assert slowest['name'] == 'python.reranker'
    text = json.dumps(exported) + json.dumps(legacy)
    for forbidden in (private, 'alex@example.test', 'PRIVATE_PROMPT_SENTINEL',
                      'trusted-execution-fixture', 'fixture-secret'):
        assert forbidden not in text
    print('Same slow ticket: delayed rerank identified as longest leaf hop; Rails trace preserved; PII absent.')
    print('Offline smoke: collector delivery, dashboard queries, live HTTP ingress and model quality not verified; release blocked.')


def test_delay_is_explicitly_test_only():
    with pytest.raises(ValueError, match='test_only_delay_required'):
        DelayedTestScorer(OfflinePairScorer())
