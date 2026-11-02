import asyncio
import json
import os
import re
from pathlib import Path

from anchor_ai.observability import span
from anchor_ai.prompts import QUERY_ANALYSIS_INSTRUCTION, QUERY_ANALYSIS_VERSION
from anchor_ai.schemas import QueryPlanPayload, RetrievalPlan

ROOT = Path(__file__).resolve().parents[1]
IDENTIFIER = re.compile(r'\b(?:SKU|ORDER)-[A-Za-z0-9]+\b|\b\d{4,}\b')


def identifiers(text):
    return set(IDENTIFIER.findall(text))


def sanitize(ticket):
    if not isinstance(ticket, str) or not ticket.strip():
        raise ValueError('invalid_query')
    if len(ticket.encode('utf-8')) > 1800:
        raise ValueError('invalid_query')
    text = re.sub(r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}',
                  '[EMAIL]', ticket, flags=re.I)
    text = re.sub(r'\b[0-9]{3}-[0-9]{3}-[0-9]{4}\b', '[PHONE]', text)
    return text.strip()


class RecordedQueryClient:
    def __init__(self):
        path = ROOT / 'anchor_ai/tests/fixtures/query_analysis_responses.json'
        self.responses = json.loads(path.read_text())
        self.calls = []

    async def complete(self, ticket):
        self.calls.append(ticket)
        return json.dumps(self.responses.get(ticket, {'queries': [ticket]}))


class ExistingQueryClient:
    def __init__(self, max_tokens=256):
        self.max_tokens = max_tokens

    async def complete(self, ticket):
        from anchor_ai.llm_client import LlmClient
        return await LlmClient().complete(
            ticket, system=QUERY_ANALYSIS_INSTRUCTION, max_tokens=self.max_tokens)


def query_settings():
    import yaml
    return yaml.safe_load((ROOT / 'config/retrieval.yml').read_text())['query_analysis']


def make_analyzer():
    settings = query_settings()
    mode = os.environ.get('ANCHOR_QUERY_PROVIDER', 'offline')
    if mode not in ('offline', 'live'):
        raise ValueError('invalid_query_provider')
    client = (RecordedQueryClient() if mode == 'offline' else
              ExistingQueryClient(settings['max_tokens']))
    return QueryAnalyzer(client, max_queries=settings['max_queries'],
                         deadline_seconds=settings['deadline_seconds'])


class QueryAnalyzer:
    def __init__(self, client, max_queries=3, deadline_seconds=1):
        if type(max_queries) is not int or not 1 <= max_queries <= 3:
            raise ValueError('invalid_query_cap')
        if not 0 < deadline_seconds <= 10:
            raise ValueError('invalid_analysis_deadline')
        self.client = client
        self.max_queries = max_queries
        self.deadline_seconds = deadline_seconds

    async def analyze(self, ticket):
        original = sanitize(ticket)
        with span('python.query_analysis', stage='query_analysis',
                  analysis_version=QUERY_ANALYSIS_VERSION) as item:
            reason = 'invalid_output'
            calls = 0
            # An identifier-only lookup has no language intent to clarify.
            if original in identifiers(original):
                plan = RetrievalPlan(version=QUERY_ANALYSIS_VERSION,
                    queries=[original], fallback=False, reason='identifier_bypass')
            else:
                try:
                    calls = 1
                    raw = await asyncio.wait_for(self.client.complete(original),
                                                 self.deadline_seconds)
                    if not isinstance(raw, str) or len(raw.encode('utf-8')) > 4096:
                        raise ValueError('invalid_output')
                    payload = QueryPlanPayload.model_validate_json(raw)
                    queries = list(dict.fromkeys(q.strip() for q in payload.queries))
                    if len(queries) > self.max_queries:
                        raise ValueError('invalid_output')
                    if not identifiers(original) <= identifiers(' '.join(queries)):
                        reason = 'identifier_loss'
                        raise ValueError(reason)
                    if any(sanitize(q) != q for q in queries):
                        raise ValueError('invalid_output')
                    plan = RetrievalPlan(version=QUERY_ANALYSIS_VERSION,
                        queries=queries, fallback=False, reason='accepted')
                except Exception as error:
                    if isinstance(error, TimeoutError):
                        reason = 'timeout'
                    elif not isinstance(error, ValueError):
                        reason = 'client_error'
                    plan = RetrievalPlan(version=QUERY_ANALYSIS_VERSION,
                        queries=[original], fallback=True, reason=reason)
            item.set(analysis_calls=calls, subquery_count=len(plan.queries),
                     analysis_fallback=plan.fallback, analysis_reason=plan.reason)
            return plan
