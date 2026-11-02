import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from anchor_ai.memory import RailsAgentMemory
from anchor_ai.orchestration import run_support_conversation

RUNNER = '''
Rails.logger = ActiveSupport::Logger.new(File::NULL)
input = JSON.parse(STDIN.read)
session = ActionDispatch::Integration::Session.new(Rails.application)
session.host! 'localhost'
if input.fetch('method') == 'POST'
  session.post('/api/agent_memories', params: input.fetch('payload'),
    headers: input.fetch('headers'), as: :json)
else
  session.get('/api/agent_memories', params: input.fetch('payload'),
    headers: input.fetch('headers'))
end
raise "memory endpoint failed: #{session.response.status}" unless session.response.status == 200
puts session.response.body
'''


def rails_request(method, payload, headers):
    env = dict(os.environ, ANCHOR_MEMORY_CREDENTIALS='{"local-lab-worker":3709}')
    completed = subprocess.run(
        ['bin/rails', 'runner', RUNNER], cwd=ROOT, env=env,
        input=json.dumps({'method': method, 'payload': payload, 'headers': headers}),
        text=True, capture_output=True, check=True, timeout=8)
    return json.loads(completed.stdout.strip().splitlines()[-1])


async def main():
    os.environ['ANCHOR_LLM_PROVIDER'] = 'offline'
    memory = RailsAgentMemory('local-lab-worker', request=rails_request)
    ticket = {'conversation_id': 123, 'order_id': 123,
              'question': 'Can this order be refunded? Contact sam@example.com.',
              'policy': 'Manual approval required.'}
    result = await run_support_conversation(ticket, memory, enabled=True)
    assert result['speakers'] == ['triage', 'refund_specialist']
    assert result['rounds'] < result['max_round']
    assert 'TERMINATE' in result['stop_reason']
    assert 'sam@example.com' not in result['summary']
    rows = memory.load(123)
    matching = [row for row in rows if row['run_key'] == 'refund-123-v1']
    assert len(matching) == 1
    assert matching[0]['summary'] == result['summary']
    print(json.dumps(result, indent=2))
    print('triage -> refund_specialist -> TERMINATE before max_round')
    print('agent_memories: one redacted durable summary for this tenant/run')
    print('restored_summaries:', result['restored_summaries'])


if __name__ == '__main__':
    asyncio.run(main())
