import argparse
import asyncio
import json
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anchor_ai.contracts import VerifiedExecution
from anchor_ai.agentic_rag import build_bounded_runtime


class OfflineModel:
    reservation = 8

    async def propose(self, strategy, plan, memory, max_tokens):
        if os.getenv('ANCHOR_LLM_PROVIDER', 'offline') != 'offline':
            raise RuntimeError('inject_a_live_accounted_model_adapter')
        if strategy == 'plan_execute':
            # Execute the saved explicit plan; observations still gate dispatch.
            action = plan[len(memory)]
        elif strategy == 'react':
            # Select the next action from observed feedback.
            observed = {m['action'] for m in memory}
            action = next(a for a in plan if a not in observed)
        else:
            raise ValueError('unknown_strategy')
        return action, 8


async def offline_retrieve(scope, message):
    return {'tenant_id': scope.tenant_id, 'generation_id': 'offline-refund-v1',
            'hits': [{'id': 901, 'tenant_id': scope.tenant_id,
                      'text': 'Eligible refunds require a dry run and human approval.'},
                     {'id': 902, 'tenant_id': scope.tenant_id,
                      'text': 'Refunds above the tenant ceiling must not execute.'}]}


class OfflineBackend:
    def __init__(self, path):
        self.path = str(path)
        with sqlite3.connect(self.path) as db:
            db.execute('CREATE TABLE IF NOT EXISTS refunds '
                       '(key TEXT PRIMARY KEY, payload TEXT, receipt TEXT)')

    async def call(self, scope, action, args, key):
        if action == 'lookup_order':
            return {'order_id': args['order_id'], 'owner_tenant': scope.tenant_id}
        if action == 'check_refund':
            return {'order_id': args['order_id'], 'eligible': True, 'dry_run': True}
        if action != 'issue_refund':
            raise ValueError('unknown_backend_action')
        payload = json.dumps([scope.tenant_id, scope.conversation_id, args], sort_keys=True)
        with sqlite3.connect(self.path) as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT payload, receipt FROM refunds WHERE key=?', (key,)).fetchone()
            if row:
                if row[0] != payload:
                    raise ValueError('idempotency_arguments_changed')
                return json.loads(row[1])
            receipt = dict(args, status='refunded', idempotency_key=key,
                           backend='offline', receipt_id='offline-' + key[:16])
            db.execute('INSERT INTO refunds VALUES(?,?,?)',
                       (key, payload, json.dumps(receipt)))
            return receipt

    def count(self):
        with sqlite3.connect(self.path) as db:
            return db.execute('SELECT count(*) FROM refunds').fetchone()[0]


def identities():
    # Explicit offline authentication double. Production verifies an opaque
    # context credential against the application identity service instead.
    return {
        'worker': VerifiedExecution(1, 104, 'support-7', 'support_supervisor'),
        'reviewer': VerifiedExecution(1, 104, 'reviewer-9', 'refund_reviewer'),
        'other': VerifiedExecution(2, 104, 'support-8', 'support_supervisor'),
        'reader': VerifiedExecution(1, 104, 'reader-1', 'support_reader'),
    }


def make_runtime(directory, model=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    backend = OfflineBackend(directory / 'backend.sqlite3')
    runtime = build_bounded_runtime(
        directory / 'runtime.sqlite3', verify=identities().get,
        model=model or OfflineModel(), retrieve=offline_retrieve, backend=backend)
    return runtime, backend


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('fixture')
    parser.add_argument('--state-dir', default='tmp/bounded_agent')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--strategy', choices=['react', 'plan_execute'])
    args = parser.parse_args()
    runtime, backend = make_runtime(args.state_dir)
    if args.strategy:
        runtime.policy['bounded']['strategies']['multi-step refund'] = args.strategy
    ticket = json.loads(Path(args.fixture).read_text())
    if args.resume:
        runtime.review('reviewer', ticket['run_id'], True)
        ticket['operation'] = 'resume'
    result = await runtime.execute('worker', ticket)
    result['backend_refund_count'] = backend.count()
    output = Path(args.state_dir) / (ticket['run_id'] + '.trace.json')
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    asyncio.run(main())
