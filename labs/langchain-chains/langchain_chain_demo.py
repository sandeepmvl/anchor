import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from langchain_core.exceptions import OutputParserException
from anchor_ai.chains import run_ticket_chain
from anchor_ai.contracts import PromptContract
from anchor_ai.llm_client import ChainReplayProvider, LlmClient


async def rails_prompt():
    environment = dict(os.environ, ANCHOR_REPLY_PROMPT_VERSION='chat:v3')
    result = subprocess.run(
        ['ruby', 'bin/rails', 'runner', 'puts JSON.generate(PromptRegistry.active_reply)'],
        cwd=ROOT, env=environment, text=True, capture_output=True,
        check=True, timeout=8)
    contract = PromptContract.model_validate_json(result.stdout.strip().splitlines()[-1])
    print(json.dumps({'registry_resolved_version': contract.version}), file=sys.stderr)
    return contract


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticket', required=True)
    parser.add_argument('--invalid', action='store_true')
    parser.add_argument('--retry', action='store_true')
    args = parser.parse_args()
    ticket = json.loads(Path(args.ticket).read_text())
    good = json.dumps({'reply': 'I can help check the return policy for order 42.',
                       'status': 'needs_review'})
    bad = json.dumps({'reply': 'Refund sent.', 'status': 'refund_sent'})
    responses = [bad, good] if args.invalid and args.retry else [bad] if args.invalid else [good]
    provider = ChainReplayProvider(responses)
    try:
        reply = await run_ticket_chain(ticket, client=LlmClient(provider),
                                       resolve_prompt=rails_prompt, retry=args.retry)
    except OutputParserException:
        print(json.dumps({'outcome': 'rejected', 'error_type': 'OutputParserException'}))
    else:
        print(reply.model_dump_json())
    print(json.dumps({'offline_model_calls': provider.calls}), file=sys.stderr)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    asyncio.run(main())
