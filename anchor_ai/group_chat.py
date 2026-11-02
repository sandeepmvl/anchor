import json

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response, TerminationCondition
from autogen_agentchat.messages import StopMessage, TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat as GroupChat

from anchor_ai.orchestration import ConversationRoleClient
from anchor_ai.pii import redact_memory

MARKER = {'status': 'advisory_complete', 'marker': 'TERMINATE'}


def completed(message):
    if not isinstance(message, TextMessage) or message.source != 'refund_specialist':
        return False
    try:
        return json.loads(message.content) == MARKER
    except (ValueError, TypeError):
        return False


class StructuredCompletion(TerminationCondition):
    def __init__(self):
        self._terminated = False

    @property
    def terminated(self):
        return self._terminated

    async def __call__(self, messages):
        if any(completed(message) for message in messages):
            self._terminated = True
            return StopMessage(content='trusted TERMINATE', source='anchor_control')
        return None

    async def reset(self):
        self._terminated = False


class ConversableAgent(BaseChatAgent):
    """Anchor role adapter for AgentChat 0.7, not a historical SDK class."""
    def __init__(self, name, client, *, emit_completion=True):
        super().__init__(name=name, description=name + ' advisory support role')
        self.client = client
        self.emit_completion = emit_completion
        self.context = []

    @property
    def produced_message_types(self):
        return (TextMessage,)

    async def on_messages(self, messages, cancellation_token):
        self.context.extend(messages)
        context = [{'source': m.source, 'content': str(m.content)}
                   for m in self.context if isinstance(m, TextMessage)]
        text = redact_memory(str(self.client.complete(self.name, context)))[:1500]
        if self.name == 'refund_specialist':
            # This adapter acknowledges delivery of the advisory response only.
            # No model or customer can authorize a payment through this marker.
            payload = dict(MARKER) if self.emit_completion else {'status': 'needs_review'}
            payload['answer'] = text
            text = json.dumps(payload)
        message = TextMessage(content=text, source=self.name)
        self.context.append(message)
        return Response(chat_message=message)

    async def on_reset(self, cancellation_token):
        self.context.clear()


# Require the exact envelope, including an advisory answer, from the specialist.
def trusted_completion(message):
    if not isinstance(message, TextMessage) or message.source != 'refund_specialist':
        return False
    try:
        payload = json.loads(message.content)
    except (ValueError, TypeError):
        return False
    return (isinstance(payload, dict)
            and set(payload) == {'status', 'marker', 'answer'}
            and payload['status'] == MARKER['status']
            and payload['marker'] == MARKER['marker']
            and isinstance(payload['answer'], str) and bool(payload['answer']))


# The framework condition and final status use the same predicate.
completed = trusted_completion


async def run_conversation(ticket, memory, *, run_key='refund-123-v1',
                           max_round=6, emit_completion=True, client=None):
    if type(max_round) is not int or not 2 <= max_round <= 6:
        raise ValueError('invalid_max_round')
    conversation_id = ticket['conversation_id']
    prior = memory.load(conversation_id)
    roles = [ConversableAgent('triage', client or ConversationRoleClient()),
             ConversableAgent('refund_specialist', client or ConversationRoleClient(),
                              emit_completion=emit_completion)]
    team = GroupChat(roles, termination_condition=StructuredCompletion(),
                     max_turns=max_round)
    task = json.dumps({'label': 'UNTRUSTED_TASK_AND_MEMORY',
                       'ticket': ticket, 'previous_summaries': prior})
    # Redaction happens before role delivery, provider calls, or durable storage.
    result = await team.run(task=TextMessage(content=redact_memory(task), source='customer'))
    replies = [m for m in result.messages if isinstance(m, TextMessage)
               and m.source in ('triage', 'refund_specialist')]
    resolved = bool(replies and completed(replies[-1]))
    status = 'advisory_complete' if resolved else 'human_escalation'
    handoff = {
        'ticket': ticket,
        'policy': 'Manual approval required; no refund executed.',
        'status': status,
        'stop_reason': result.stop_reason,
        'public_conversation': [{'role': m.source, 'content': m.content} for m in replies],
        'next_action': 'authenticated approval review' if resolved else 'human investigation',
    }
    safe = redact_memory(json.dumps(handoff, ensure_ascii=False))
    saved = memory.save(conversation_id, run_key, safe)
    return {'status': status, 'speakers': [m.source for m in replies],
            'rounds': len(replies), 'max_round': max_round,
            'stop_reason': result.stop_reason, 'summary': saved['summary'],
            'restored_summaries': len(prior)}
