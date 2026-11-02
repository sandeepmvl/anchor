import itertools
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from anchor_ai.embeddings import FIXTURE_PATH, cosine, dot
from anchor_ai.llm_client import LlmClient


def main():
    fixture = json.loads(FIXTURE_PATH.read_text())
    texts = [record['text'] for record in fixture['records']]
    client = LlmClient(provider='offline')
    print('Existing offline generation:', client.complete(texts[0]))
    documents = client.embed(texts, provider='offline')
    queries = client.embed(texts, input_type='query', provider='offline')
    assert documents.vectors == queries.vectors
    assert documents.vectors == client.embed(texts, provider='offline').vectors
    print('Embedding provider:', documents.provider)
    print('Model:', documents.model, 'revision:', documents.revision,
          'dimensions:', documents.dimensions)
    print('Provenance:', fixture['provenance'])
    for index, (text, vector) in enumerate(zip(texts, documents.vectors)):
        print(index, text, '=>', vector)
    for i, j in itertools.combinations(range(len(texts)), 2):
        print(f'cosine({i}, {j}) = {cosine(documents.vectors[i], documents.vectors[j]):.6f}')
    a, b = documents.vectors[:2]
    unit_a = [x / math.sqrt(dot(a, a)) for x in a]
    unit_b = [x / math.sqrt(dot(b, b)) for x in b]
    assert math.isclose(cosine(a, b), dot(unit_a, unit_b))
    assert math.isclose(cosine(a, b), cosine([7 * x for x in a], b))
    print('Repeatable vectors, both input modes, and unit-vector dot product verified.')


if __name__ == '__main__':
    main()
