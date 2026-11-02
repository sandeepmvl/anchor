"""Release adapter and fresh parity evaluation through Anchor's /rag route."""
import argparse
import asyncio
import copy
import json
import os
import uuid
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from anchor_ai.rag import answer_release
from eval.heuristic_judge import score_capture, write_json
from eval.triad import fingerprint

ROOT = Path(__file__).resolve().parents[1]


def native(value):
    if isinstance(value, dict):
        return {key: native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    if hasattr(value, 'item'):
        return native(value.item())
    return value


class ReleaseRuntime:
    def __init__(self, dataset=None, client=None, search_factory=None, degraded=False):
        if dataset is None:
            dataset = json.loads((ROOT / 'eval/datasets/rag_thresholds.json').read_text())
        self.dataset = copy.deepcopy(dataset)
        self.client = client
        self.search_factory = search_factory
        self.degraded = degraded
        self.tickets = {str(ticket['id']): ticket for ticket in self.dataset['tickets']}
        if len(self.tickets) != len(self.dataset['tickets']):
            raise ValueError('duplicate_release_ticket')
        if search_factory is None:
            if os.getenv('ANCHOR_LLM_PROVIDER', 'offline') != 'offline':
                raise ValueError('live_release_requires_search_factory')
            if self.dataset['configuration']['provider'] != 'offline':
                raise ValueError('recorded_search_requires_offline_configuration')

    async def answer(self, body):
        ticket = self.tickets.get(body.ticket_id)
        if ticket is None:
            raise ValueError('unknown_release_ticket')
        if (ticket['tenant_id'] != body.tenant_id or
                ticket['generation_id'] != body.generation_id or
                ticket['question'] != body.question):
            raise ValueError('release_scope_mismatch')
        if self.search_factory is None:
            from eval.rag_gate import offline_search
            search = offline_search(self.dataset['chunks'], ticket, self.degraded)
        else:
            search = self.search_factory(ticket)
        row = await answer_release(copy.deepcopy(ticket),
            self.dataset['configuration'], search, client=self.client)
        if (row['tenant_id'] != body.tenant_id or
                row['generation_id'] != body.generation_id):
            raise ValueError('result_scope_mismatch')
        return native(dict(scope=dict(tenant_id=body.tenant_id,
            conversation_id=body.conversation_id, generation_id=body.generation_id),
            capture=row))


def request_for(ticket):
    return dict(ticket_id=str(ticket['id']), tenant_id=ticket['tenant_id'],
        conversation_id=ticket.get('conversation_id', 10),
        generation_id=ticket['generation_id'], question=ticket['question'])


def collect(client, dataset, secret):
    rows = []
    for ticket in dataset['tickets']:
        body = request_for(ticket)
        response = client.post('/rag', json=body,
            headers={'X-Anchor-Secret': secret})
        response.raise_for_status()
        payload = response.json()
        expected_scope = {key: body[key] for key in
            ('tenant_id', 'conversation_id', 'generation_id')}
        if payload['scope'] != expected_scope:
            raise ValueError('endpoint_scope_changed')
        rows.append(payload['capture'])
    return rows


def evaluate(output, dataset=None, client=None, degraded=False, base_url=None):
    from anchor_ai.app import app
    from eval.rag_gate import capture, git, load_dataset
    dataset = copy.deepcopy(dataset or load_dataset())
    output = Path(output)
    provenance = dict(run_id=uuid.uuid4().hex, commit_sha=git('rev-parse', 'HEAD'),
        config_hash=fingerprint(dataset['configuration']),
        dataset_hash=fingerprint(dataset), flag=dataset['configuration']['advanced_rag'])
    local = native(asyncio.run(capture(dataset, provenance, client, degraded)))
    if base_url:
        secret = os.environ['ANCHOR_AI_SECRET']
        with httpx.Client(base_url=base_url, timeout=30.0) as transport:
            rows = collect(transport, dataset, secret)
    else:
        secret = 'endpoint-parity-test-only'
        factory = lambda: ReleaseRuntime(dataset, client=client, degraded=degraded)
        with patch.dict(os.environ, {'ANCHOR_AI_SECRET': secret}), patch.object(
                app.state, 'rag_release_runtime_factory', factory, create=True):
            with TestClient(app) as transport:
                if transport.get('/healthz').json() != {'ready': True}:
                    raise ValueError('endpoint_not_ready')
                rows = collect(transport, dataset, secret)
    endpoint = dict(provenance=provenance, rows=rows)
    # Persist both captures even if scoring subsequently rejects their contents.
    write_json(output / 'local-capture.json', local)
    write_json(output / 'endpoint-capture.json', endpoint)
    local_scores = score_capture(local)
    endpoint_scores = score_capture(endpoint)
    write_json(output / 'local-scores.json', local_scores)
    write_json(output / 'endpoint-scores.json', endpoint_scores)
    passed = endpoint == local and endpoint_scores == local_scores
    report = dict(passed=passed, provenance=provenance,
        local_hash=fingerprint(local), endpoint_hash=fingerprint(endpoint),
        error=None if passed else 'endpoint_parity_failed')
    write_json(output / 'report.json', report)
    if not passed:
        raise ValueError('endpoint_parity_failed')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--testclient', action='store_true')
    mode.add_argument('--base-url')
    parser.add_argument('--output', default='.git/anchor-endpoint-eval')
    args = parser.parse_args(argv)
    try:
        report = evaluate(args.output, base_url=args.base_url)
    except Exception as error:
        print('BLOCK: endpoint evaluation: ' + str(error))
        return 1
    print(json.dumps(report, sort_keys=True))
    print('PASS: Anchor /rag preserves the refund answer, evidence and quality scores')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
