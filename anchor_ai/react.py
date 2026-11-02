import asyncio
import hashlib
import json

from anchor_ai.agentic_rag import ABSTENTION, AgenticRuntime
from anchor_ai.autonomy import run_for
from anchor_ai.contracts import ToolScope
from anchor_ai.observability import AGENT_SUMMARIES, emit_react, span
from anchor_ai.schemas import AgentDecision, ReActContent, RetrieveArguments
from anchor_ai.schemas import normalize_retrieval_query


class NativePolicyAdapter:
    """Reuse decide(question, evidence); exchange native messages internally."""
    def __init__(self, policy):
        self.policy = policy

    async def complete(self, messages):
        question = messages[0]['content']
        evidence = {}
        for message in messages:
            if message['role'] == 'tool':
                for passage in json.loads(message['content'])['passages']:
                    evidence[passage['id']] = passage
        decision = AgentDecision.model_validate(
            await self.policy.decide(question, list(evidence.values())))
        payload = {'reasoning': AGENT_SUMMARIES[decision.kind]}
        message = {'role': 'assistant'}
        if decision.kind == 'retrieve':
            message['tool_calls'] = [{
                'id': 'call_' + str(len(messages)), 'type': 'function',
                'function': {'name': 'retrieve', 'arguments':
                             json.dumps(decision.arguments.model_dump())}}]
        else:
            payload['final'] = decision.answer.model_dump()
        message['content'] = json.dumps(payload)
        return message


class ReActRuntime(AgenticRuntime):
    """Read-only ReAct strategy; consequential tools are not enabled here."""
    def __init__(self, retriever, *, native_client=None, detect_cycles=True, **kwargs):
        super().__init__(retriever, **kwargs)
        self.native_client = native_client or NativePolicyAdapter(self.policy)
        self.detect_cycles = detect_cycles
        self.last_exit = None

    async def answer(self, question, k=3, *, request_id, trace_id,
                     conversation_seen_ids=(), execution_scope=None, execution_run=None):
        if not isinstance(question, str) or not question.strip() or len(question.encode('utf-8')) > 1800:
            raise ValueError('current_question_budget_exceeded')
        if type(k) is not int or not 1 <= k <= 20:
            raise ValueError('invalid_top_k')
        history = tuple(conversation_seen_ids)
        if any(type(i) is not int or i <= 0 for i in history):
            raise ValueError('invalid_conversation_seen_ids')
        if history and self.tool.generation_id is None:
            raise ValueError('agent_scope_mismatch')
        scope = execution_scope or ToolScope(tenant_id=self.tool.tenant_id, conversation_id=1)
        if scope.tenant_id != self.tool.tenant_id:
            raise ValueError('agent_scope_mismatch')
        state = run_for(scope, execution_run)
        messages = [{'role': 'user', 'content': question}]
        evidence, signatures, call_ids = {}, set(), set()
        calls = 0
        self.last_trajectory = []
        self.last_exit = None
        ctx = {'request_id': request_id, 'trace_id': trace_id, 'parent_span_id': None}
        with span('python.react', ctx=ctx, stage='agent',
                  tenant_id=scope.tenant_id, agent_mode=self.mode) as root:
            def record(event, **fields):
                def export(row):
                    self.last_trajectory.append(row)
                    self.emit(row)
                emit_react(event, export, **fields)

            def finish(reason, answer=None):
                state.halted = True
                self.last_exit = reason
                result = answer or {'reply': ABSTENTION, 'citations': [], 'abstained': True}
                root.set(agent_exit=reason, tool_calls=calls,
                         evidence_chunk_ids=sorted(evidence),
                         citations=result['citations'], abstained=result['abstained'])
                record('exit', agent_exit=reason, tool_calls=calls,
                       evidence_chunk_ids=sorted(evidence))
                return result

            for iteration in range(1, self.max_iterations + 1):
                reason = await state.step(scope)
                if reason:
                    return finish(reason)
                with span('python.react.model', stage='llm',
                          agent_iteration=iteration,
                          iterations_remaining=self.max_iterations - iteration) as model_span:
                    try:
                        message = await asyncio.wait_for(
                            self.native_client.complete(messages), self.policy_timeout)
                        if message['role'] != 'assistant':
                            raise ValueError('invalid_role')
                        content = ReActContent.model_validate_json(message['content'])
                        proposed = message.get('tool_calls', [])
                        if not isinstance(proposed, list) or len(proposed) > 1:
                            raise ValueError('one_call_per_turn')
                        if proposed:
                            if content.final is not None or content.reasoning != AGENT_SUMMARIES['retrieve']:
                                raise ValueError('invalid_action_content')
                            call = proposed[0]
                            call_id = call['id']
                            if not isinstance(call_id, str) or not 1 <= len(call_id) <= 80 or call_id in call_ids:
                                raise ValueError('invalid_call_id')
                            if call['type'] != 'function' or call['function']['name'] != 'retrieve':
                                raise ValueError('unknown_tool')
                            args = RetrieveArguments.model_validate_json(call['function']['arguments'])
                        elif content.final is None or content.reasoning == AGENT_SUMMARIES['retrieve']:
                            raise ValueError('missing_final')
                    except (ValueError, TypeError, KeyError):
                        model_span.fail('agent_invalid_decision')
                        return finish('invalid_decision')
                    except Exception:
                        model_span.fail('invalid_provider_response')
                        return finish('policy_error')
                    model_span.set(decision_summary=content.reasoning)
                record('decision_summary', agent_iteration=iteration,
                       decision_summary=content.reasoning, tool_calls=calls)
                if not proposed:
                    answer = content.final.model_dump()
                    citations = answer['citations']
                    if (len(citations) != len(set(citations))
                            or any(type(i) is not int or i not in evidence for i in citations)
                            or (not answer['abstained'] and not citations)):
                        return finish('invalid_answer')
                    return finish('final_answer', answer)
                normalized = normalize_retrieval_query(args.query)
                fingerprint = hashlib.sha256(('retrieve:' + normalized).encode()).hexdigest()
                record('action', agent_iteration=iteration, tool_name='retrieve',
                       query_fingerprint=fingerprint, tool_calls=calls)
                if self.detect_cycles and fingerprint in signatures:
                    return finish('repeated_action')
                if calls >= self.max_tool_calls:
                    return finish('tool_budget')
                if calls >= self.max_hops:
                    return finish('max_hops')
                reason, _ = await state.admit(scope, 'retrieve', args.model_dump())
                if reason:
                    return finish(reason)
                signatures.add(fingerprint)
                call_ids.add(call_id)
                calls += 1
                with span('python.react.tool', stage='tool', tool_name='retrieve',
                          agent_iteration=iteration, query_fingerprint=fingerprint,
                          tool_calls=calls) as tool_span:
                    try:
                        observation = await self.tool(args.model_dump(), k)
                    except Exception:
                        tool_span.fail('agent_tool_failed')
                        return finish('tool_error')
                for passage in observation['passages']:
                    candidate = dict(evidence)
                    candidate[passage['id']] = passage
                    size = len(json.dumps(list(candidate.values()), ensure_ascii=False).encode('utf-8'))
                    if size <= self.evidence_budget:
                        evidence = candidate
                # Send only retained evidence to the next invocation. Bound the
                # transcript by iteration count and the existing evidence cap.
                retained = {'passages': [p for p in observation['passages']
                                         if p['id'] in evidence]}
                messages.extend([message, {'role': 'tool', 'tool_call_id': call_id,
                                          'content': json.dumps(retained)}])
                record('observation', agent_iteration=iteration, tool_name='retrieve',
                       tool_calls=calls, evidence_chunk_ids=sorted(evidence),
                       observation_redacted=True)
            return finish('iteration_budget')
