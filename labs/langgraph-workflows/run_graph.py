import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from anchor_ai.checkpoints import CheckpointStore, actor_claims, environment_token
from anchor_ai.graphs import resume_refund, start_refund


def approve_via_rails(thread, token, path):
    env = dict(os.environ)
    env['DATABASE_URL'] = 'sqlite3:' + str(Path(path).resolve())
    env['GRAPH_THREAD'] = thread
    env['GRAPH_TOKEN'] = token
    script = '''
      require "json"
      session = ActionDispatch::Integration::Session.new(Rails.application)
      session.host! "localhost"
      headers = { "Authorization" => "Bearer #{ENV.fetch('GRAPH_TOKEN')}" }
      base = "/api/agent_checkpoints/#{ENV.fetch('GRAPH_THREAD')}"
      session.get(base, headers: headers)
      raise "checkpoint read denied" unless session.response.status == 200
      key = JSON.parse(session.response.body).fetch("action_key")
      session.post(base + "/approve", params: { action_key: key }, headers: headers, as: :json)
      raise "approval denied or replayed" unless session.response.status == 201
      puts session.response.body
    '''
    subprocess.run(['bin/rails', 'runner', script], cwd=ROOT, env=env, check=True)


def select_thread(store, requested, token):
    if requested != 'latest':
        return requested
    claims = actor_claims(token)
    with store.db() as db:
        row = db.execute('''SELECT thread_id FROM agent_checkpoints
            WHERE tenant_id=? ORDER BY id DESC LIMIT 1''',
            (claims['tenant_id'],)).fetchone()
    if row is None:
        raise ValueError('no_checkpoint_for_tenant')
    return row['thread_id']


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--storage', action='store_true')
    group.add_argument('--ticket')
    group.add_argument('--approve')
    group.add_argument('--resume')
    group.add_argument('--inspect')
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--db', default=os.getenv('ANCHOR_GRAPH_DB', 'storage/development.sqlite3'))
    args = parser.parse_args()
    if args.demo:
        # Explicit local exercise mode, never an implicit authentication fallback.
        os.environ['ANCHOR_GRAPH_DEMO'] = '1'
        os.environ['ANCHOR_LLM_PROVIDER'] = 'fake'
        os.environ.setdefault('ANCHOR_APPROVAL_SECRET',
                              'local-only-approval-secret-at-least-32-bytes')
    store = CheckpointStore(args.db)
    if args.storage:
        print(store.available())
        return
    token = environment_token()
    if args.ticket:
        thread = start_refund(args.db, json.loads(Path(args.ticket).read_text()), token)
        print('thread_id=' + thread)
        print('PAUSED at human_approval_checkpoint; process exits now.')
        return
    thread = select_thread(store, args.approve or args.inspect or args.resume, token)
    if args.approve:
        approve_via_rails(thread, token, args.db)
        print('Approval recorded against the pending action; graph has not resumed.')
    elif args.inspect:
        row = store.inspect(thread, token)
        print(json.dumps({'thread_id': row['thread_id'], 'status': row['status'],
                          'context': json.loads(row['context_json'])}, sort_keys=True))
    else:
        values = resume_refund(args.db, thread, token)
        print('RESUMED original context after process restart; triage called once.')
        print('Completed tool actions protected by durable idempotency keys; no money moved.')
        print(json.dumps({'audit': values['audit'], 'reply': values['reply']}, sort_keys=True))


if __name__ == '__main__':
    main()
