import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / 'config/retrieval.yml'
FIXTURE_PATH = Path(__file__).resolve().parent / 'tests/fixtures/embedding_responses.json'


@dataclass(frozen=True)
class EmbeddingBatch:
    vectors: list[list[float]]
    provider: str
    model: str
    revision: str
    dimensions: int
    input_type: str


def load_config():
    return yaml.safe_load(CONFIG_PATH.read_text())['embeddings']


def validate_config(provider, settings):
    model, dimensions = settings['model'], settings['dimensions']
    if type(dimensions) is not int or dimensions <= 0:
        raise ValueError('dimensions must be a positive integer')
    if not isinstance(settings['revision'], str) or not settings['revision']:
        raise ValueError('a configuration revision is required')
    if provider == 'offline':
        if (model, dimensions, settings['revision']) != (
                'anchor-recorded-embedding-v1', 4, 'authored-fixture-v1'):
            raise ValueError('incompatible offline model configuration')
    elif provider == 'openai':
        maximum = {'text-embedding-3-small': 1536, 'text-embedding-3-large': 3072}
        if model not in maximum or dimensions > maximum[model]:
            raise ValueError('incompatible OpenAI model dimensions')
    elif provider == 'voyage':
        # This adapter supports the selected default output size only.
        if model != 'voyage-3.5' or dimensions != 1024:
            raise ValueError('incompatible Voyage model dimensions')
    else:
        raise ValueError('provider must be offline, openai, or voyage')


def validate_vectors(vectors, count, dimensions):
    if len(vectors) != count:
        raise ValueError('embedding response count mismatch')
    for vector in vectors:
        if not isinstance(vector, list) or len(vector) != dimensions:
            raise ValueError('embedding response dimension mismatch')
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in vector):
            raise ValueError('embedding coordinates must be finite numbers')
        if not any(vector):
            raise ValueError('zero vectors have no cosine direction')


def ordered_vectors(body, count):
    data = body['data']
    indices = [item['index'] for item in data]
    if any(type(index) is not int for index in indices):
        raise ValueError('embedding indices must be integers')
    if sorted(indices) != list(range(count)):
        raise ValueError('embedding indices must cover each input exactly once')
    return [item['embedding'] for item in sorted(data, key=lambda item: item['index'])]


def embed(texts, input_type='document', *, provider=None, config=None,
          transport=None, api_key=None):
    if input_type not in {'query', 'document'}:
        raise ValueError('input_type must be query or document')
    if not isinstance(texts, list) or not texts:
        raise ValueError('texts must be a nonempty list')
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError('each text must be a nonempty string')
    config = load_config() if config is None else config
    provider = provider or os.environ.get('ANCHOR_EMBED_PROVIDER') or config['default_provider']
    if provider not in config['providers']:
        raise ValueError('unknown embedding provider')
    settings = config['providers'][provider]
    validate_config(provider, settings)
    model, dimensions = settings['model'], settings['dimensions']
    if provider == 'offline':
        fixture = json.loads(FIXTURE_PATH.read_text())
        if (fixture['model'], fixture['revision'], fixture['dimensions']) != (
                model, settings['revision'], dimensions):
            raise ValueError('fixture configuration mismatch')
        lookup = {record['text']: record['vector'] for record in fixture['records']}
        try:
            vectors = [list(lookup[text]) for text in texts]
        except KeyError:
            raise ValueError('text is not in the offline embedding recording') from None
    else:
        key_name = 'OPENAI_API_KEY' if provider == 'openai' else 'VOYAGE_API_KEY'
        key = api_key if api_key is not None else os.environ.get(key_name)
        if not key:
            raise ValueError(key_name + ' is required for the selected live provider')
        payload = {'model': model, 'input': texts}
        if provider == 'openai':
            url = 'https://api.openai.com/v1/embeddings'
            payload.update(dimensions=dimensions, encoding_format='float')
        else:
            url = 'https://api.voyageai.com/v1/embeddings'
            payload.update(input_type=input_type, truncation=False)
        with httpx.Client(transport=transport, timeout=10.0) as client:
            response = client.post(url, headers={'Authorization': 'Bearer ' + key}, json=payload)
            response.raise_for_status()
            try:
                body = response.json()
                if body.get('model', model) != model:
                    raise ValueError('unexpected embedding response model')
                vectors = ordered_vectors(body, len(texts))
            except (KeyError, TypeError, IndexError):
                raise ValueError('invalid embedding provider response') from None
    validate_vectors(vectors, len(texts), dimensions)
    return EmbeddingBatch(vectors, provider, model, settings['revision'], dimensions, input_type)


def dot(a, b):
    if not a or len(a) != len(b):
        raise ValueError('vectors must have equal nonzero dimensions')
    return sum(x * y for x, y in zip(a, b))


def cosine(a, b):
    denominator = math.sqrt(dot(a, a)) * math.sqrt(dot(b, b))
    if denominator == 0:
        raise ValueError('zero vectors have no cosine direction')
    return dot(a, b) / denominator


async def embed_document_batched(runtime, token, *, tenant_id, document_id,
                                  text, model, revision, input_type='document'):
    """Ingestion seam; runtime verifies tenant authority before queuing text."""
    from anchor_ai.batching import BatchEnvelope, BatchItem
    item = BatchItem(request_id=str(document_id), tenant_id=tenant_id,
                     operation='embedding', model=model, revision=revision,
                     settings={'input_type': input_type}, text=text)
    result = await runtime.execute(token, BatchEnvelope(items=[item]))
    return result['items'][0]
