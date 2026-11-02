import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from anchor_ai.contracts import PromptContract
from anchor_ai.prompts import render_chat

MANIFEST = ROOT / 'eval/datasets/regression_manifest_v2.json'
REPLAYS = ROOT / 'eval/fixtures/prompt_version_replays.jsonl'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def load_manifest():
    value = json.loads(MANIFEST.read_text())
    ids = [row['id'] for row in value['examples']]
    if len(ids) != 20 or len(set(ids)) != 20:
        raise ValueError('twenty_unique_examples_required')
    return value


def fixture_contract(tag):
    # Test input only. Production exports come from PromptRegistry.export_regression.
    templates = {
        'chat:v1': 'Input: one nonempty support ticket: {{ticket}}. Output: a plain reply string without a prefix.',
        'chat:v2': 'Input: one nonempty support ticket: {{ticket}}. Output: a reply string beginning with Support:.'}
    return dict(version=tag, template=templates[tag],
        model_id='gpt-4.1-2025-04-14', temperature=0.0, max_tokens=120)


def envelope(contract, version_id):
    contract = PromptContract.model_validate(contract).model_dump()
    return dict(tag=contract['version'], version_id=str(version_id),
                contract=contract, sha256=digest(contract))


def fixture_input():
    manifest = load_manifest()
    return dict(source='constructed_replay', manifest=manifest,
        prompts=[envelope(fixture_contract(tag), 'fixture:' + tag)
                 for tag in ('chat:v1', 'chat:v2')])


class ReplayClient:
    def __init__(self):
        if os.getenv('ANCHOR_REGRESSION_PROVIDER', 'offline') != 'offline':
            raise RuntimeError('hosted_adapter_not_implemented')
        self.calls = 0
        self.rows = {row['variant']: row for row in
                     map(json.loads, REPLAYS.read_text().splitlines())}

    def complete(self, contract, example, regressed=False):
        contract = PromptContract.model_validate(contract)
        rendered = render_chat(contract, example['query'])
        if example['query'] not in rendered:
            raise ValueError('ticket_not_rendered')
        self.calls += 1
        variant = 'baseline' if contract.version == 'chat:v1' else 'candidate'
        if regressed and variant == 'candidate' and example['id'] == 'refund-01':
            variant = 'regressed'
        return dict(self.rows[variant])


def call_api(prompt, options, context):
    config = options.get('config', {})
    tag = config.get('tag', 'chat:v1')
    variables = context['vars']
    selected = fixture_contract(tag)
    result = ReplayClient().complete(selected,
        {'id': variables['id'], 'query': variables['query']},
        regressed=config.get('regressed', False))
    return {'output': result['text'], 'metadata': {
        'evidence': 'constructed_replay', 'prompt_tag': tag,
        'contract_hash': digest(selected), 'model_version': selected['model_id'],
        'rubric_source': result['rubric_source']}}


def get_tests():
    manifest = load_manifest()
    return [{'vars': dict(row, expected=manifest['expected'])}
            for row in manifest['examples']]
