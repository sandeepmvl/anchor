import asyncio
import copy
import inspect
from contextvars import ContextVar

from anchor_ai.observability import span

ACTIVE_HOOK = ContextVar('anchor_harness_hook', default=None)


async def policy_hook(runtime, scope, state, action):
    # Application authorization remains authoritative, including on resume.
    return runtime.allowed(scope, action, state['arguments'])


async def before_dispatch(runtime, scope, state, action):
    active = ACTIVE_HOOK.get()
    if active is None:
        return True  # Preserve existing application-mode execution.
    hook, timeout = active
    decision, reason = False, 'policy_denied'
    call_id = f"{state['run_id']}_{state['phase']}"
    with span('python.harness.pre_tool', stage='agent',
              execution_mode='local_harness', call_id=call_id,
              tenant_id=scope.tenant_id, conversation_id=scope.conversation_id) as item:
        try:
            async def invoke():
                answer = hook(runtime, scope, copy.deepcopy(state), action)
                return await answer if inspect.isawaitable(answer) else answer
            answer = await asyncio.wait_for(invoke(), timeout=timeout)
            decision = answer is True
            reason = 'explicit_allow' if decision else 'policy_denied'
        except Exception:
            reason = 'policy_error'
        item.set(permission_decision='allow' if decision else 'deny',
                 permission_reason=reason)
        runtime.event(state, 'permission_decision', call_id=call_id,
                      action=action, decision='allow' if decision else 'deny',
                      reason=reason, execution_mode='local_harness')
        runtime.save(state)
    return decision


class HarnessAdapter:
    """Anchor-owned local harness integration; no third-party SDK claim."""
    def __init__(self, runtime, *, hook=policy_hook, timeout=0.2):
        self.runtime = runtime
        self.hook = hook
        self.timeout = timeout

    async def run(self, token, request):
        # Authenticate before opening any tenant-bearing span or reading state.
        scope = self.runtime.identity(token)
        if ACTIVE_HOOK.get() is not None:
            raise ValueError('nested_harness_not_supported')
        active = ACTIVE_HOOK.set((self.hook, self.timeout))
        try:
            with span('python.harness.local_loop', stage='agent',
                      execution_mode='local_harness',
                      tenant_id=scope.tenant_id,
                      conversation_id=scope.conversation_id):
                return await self.runtime.execute(token, request)
        finally:
            ACTIVE_HOOK.reset(active)


def dispatch_specialist(context, trusted, invoke, *, boundary='dispatch'):
    from .orchestration import dispatch_specialist as guarded
    return guarded(context, trusted, invoke, boundary=boundary)
