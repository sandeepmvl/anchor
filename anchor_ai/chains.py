import json
import logging
from typing import Literal

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, ConfigDict, Field

from .contracts import PromptContract
from .llm_client import LlmClient
from .observability import span
from .prompts import contract_fingerprint, resolve_reply_prompt

logger = logging.getLogger('anchor.ticket_chain')


class TicketResponse(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    reply: str = Field(min_length=1, max_length=4000)
    status: Literal['needs_review', 'answered']


class StageLogs(BaseCallbackHandler):
    def __init__(self, version, trace=None):
        self.registry_version = version
        self.version = version.rsplit(':', 1)[-1]
        self.trace = {key: value for key, value in (trace or {}).items()
                      if key in ('request_id', 'trace_id')}
        self.names = {}

    def emit(self, stage, outcome, run_id, parent_run_id, error=None):
        record = dict(self.trace, stage=stage, outcome=outcome,
                      prompt_version=self.version, registry_version=self.registry_version,
                      run_id=str(run_id),
                      parent_run_id=str(parent_run_id) if parent_run_id else None)
        if error is not None:
            record['error_type'] = type(error).__name__
        logger.info(json.dumps(record, sort_keys=True))

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
        name = kwargs.get('name') or (serialized or {}).get('name', 'runnable')
        self.names[run_id] = name
        self.emit(name, 'start', run_id, parent_run_id)

    def on_chain_end(self, outputs, *, run_id, parent_run_id=None, **kwargs):
        name = self.names.pop(run_id, 'runnable')
        self.emit(name, 'parse_success' if name == 'parser' else 'success',
                  run_id, parent_run_id)

    def on_chain_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        name = self.names.pop(run_id, 'runnable')
        outcome = ('parse_failure' if name == 'parser' and
                   isinstance(error, OutputParserException) else 'error')
        self.emit(name, outcome, run_id, parent_run_id, error)


async def run_ticket_chain(ticket, *, client=None, resolve_prompt=None,
                           retry=False, fallback_client=None, trace=None):
    """The runtime and CLI share this asynchronous, validated entry point."""
    resolver = resolve_prompt or resolve_reply_prompt
    contract = PromptContract.model_validate(await resolver())
    message = ticket['message']
    if not isinstance(message, str) or not message.strip():
        raise ValueError('invalid_ticket')
    if ('{{ticket}}' not in contract.template or
            '{{format_instructions}}' not in contract.template):
        raise ValueError('invalid_chain_prompt')
    output_parser = PydanticOutputParser(pydantic_object=TicketResponse)
    parser = output_parser.with_config(run_name='parser')
    template = contract.template.replace('{{ticket}}', '{ticket}').replace(
        '{{format_instructions}}', '{format_instructions}')
    prompt = PromptTemplate.from_template(template).partial(
        format_instructions=output_parser.get_format_instructions()
    ).with_config(run_name='registry_prompt')

    def generator(adapter, name):
        async def complete(prompt_value):
            return await adapter.complete_chain_text(prompt_value.to_string(), contract)
        return RunnableLambda(complete).with_config(run_name=name)

    candidate = generator(client if client is not None else LlmClient(), 'model') | parser
    if retry:
        candidate = candidate.with_retry(
            retry_if_exception_type=(OutputParserException,),
            stop_after_attempt=2, wait_exponential_jitter=False)
    if fallback_client is not None:
        secondary = generator(fallback_client, 'secondary_model') | parser
        candidate = candidate.with_fallbacks(
            [secondary], exceptions_to_handle=(OutputParserException,))
    chain = (prompt | candidate).with_config(run_name='ticket_chain')
    callback = StageLogs(contract.version, trace)
    with span('chain.reply', stage='chain', prompt_version=contract.version,
              prompt_fingerprint=contract_fingerprint(contract)):
        return await chain.ainvoke({'ticket': message}, config={
            'callbacks': [callback],
            'metadata': {'prompt_version': contract.version}})


async def run_policy_triage(payload, envelope, **options):
    from .orchestration import validate_framework_envelope
    validate_framework_envelope(envelope)
    trace = {'trace_id': envelope['trace_id']}
    if 'request_id' in envelope:
        trace['request_id'] = envelope['request_id']
    result = await run_ticket_chain(payload, trace=trace, **options)
    return result.model_dump()
