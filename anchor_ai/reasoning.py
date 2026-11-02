from typing import get_args

try:
    from .contracts import ReasoningEffort
except ImportError:
    from contracts import ReasoningEffort

REPLAY_MODEL = 'anchor-reasoning-replay'
CAPABILITIES = {
    REPLAY_MODEL: {'off': 'none', 'low': 'low', 'high': 'high'},
}


class ReasoningError(ValueError):
    def __init__(self, code):
        self.code = code
        self.status = 422
        super().__init__(code)


def request_fields(model, effort, max_tokens, *, offline=False):
    if effort is None:
        return {}
    if not isinstance(effort, str) or effort not in get_args(ReasoningEffort):
        raise ReasoningError('invalid_reasoning_effort')
    if type(max_tokens) is not int or not 1 <= max_tokens <= 32768:
        raise ReasoningError('invalid_reasoning_limit')
    if model == REPLAY_MODEL and not offline:
        raise ReasoningError('unsupported_reasoning_model')
    supported = CAPABILITIES.get(model, {})
    if effort not in supported:
        raise ReasoningError('unsupported_reasoning_model')
    return {'reasoning_effort': supported[effort],
            'max_completion_tokens': max_tokens}
