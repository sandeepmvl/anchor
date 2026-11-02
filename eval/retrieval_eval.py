import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from eval.retrieval_metrics import document_ids, mean_scores, score

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / 'eval/datasets/advanced_retrieval_queries.json'
TOPICS = [
    {'refund', 'refundable'},
    {'delivery', 'shipping', 'tracking'},
    {'invoice', 'receipt', 'tax'},
    {'password', 'reset', 'link'},
    {'mfa', 'authenticator', 'recovery'},
    {'cancel', 'cancellation', 'subscription'},
    {'export', 'csv'},
    {'seats', 'agent'},
    {'webhook', 'signature', 'event'},
]


def tokens(text):
    return re.findall(r'[a-z0-9]+', text.lower())


class FixtureEmbeddingClient:
    def embed(self, texts):
        provider = os.environ.get('ANCHOR_EMBEDDING_PROVIDER', 'offline')
        if provider == 'openai':
            from openai import OpenAI
            response = OpenAI().embeddings.create(
                model='text-embedding-3-small', input=texts)
            return np.array([row.embedding for row in response.data])
        if provider != 'offline':
            raise ValueError('unsupported_embedding_provider')
        vectors = []
        for text in texts:
            words = set(tokens(text))
            vector = np.zeros(64)
            for index, topic in enumerate(TOPICS):
                vector[index] = float(bool(words & topic))
            # Identifier tokens are deliberately absent from fake embeddings.
            if not vector.any():
                for word in sorted(words):
                    if not any(char.isdigit() for char in word):
                        bucket = int(hashlib.sha256(word.encode()).hexdigest(), 16)
                        vector[9 + bucket % 55] += 1
            vectors.append(vector)
        return np.array(vectors)


def normalized(matrix):
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


class FixtureRetrievalClient:
    """Evaluation adapter, not a replacement for Anchor's serving runtime."""
    def __init__(self, documents):
        self.documents = documents
        self.embeddings = FixtureEmbeddingClient()

    def retrieve_pair(self, case, depth):
        docs = sorted((d for d in self.documents
                       if d['tenant_id'] == case['tenant_id']),
                      key=lambda d: d['document_id'])
        texts = [doc['text'] for doc in docs]
        vectors = normalized(self.embeddings.embed(texts + [case['query']]))
        dense = vectors[:-1] @ vectors[-1]
        lexical = np.maximum(BM25Okapi([tokens(t) for t in texts]).get_scores(
            tokens(case['query'])), 0)
        # Fuse positions rather than adding incomparable raw score scales.
        orders = {}
        for name, values in (('dense', dense), ('lexical', lexical)):
            orders[name] = sorted(
                range(len(docs)),
                key=lambda i: (-float(values[i]), docs[i]['document_id']))
        fused = np.zeros(len(docs))
        for order in orders.values():
            for rank, index in enumerate(order, 1):
                fused[index] += 1 / (1 + rank)
        orders['hybrid'] = sorted(
            range(len(docs)),
            key=lambda i: (-float(fused[i]), docs[i]['document_id']))
        result = {}
        for name in ('dense', 'hybrid'):
            order = orders[name]
            result[name] = [dict(tenant_id=docs[i]['tenant_id'],
                                 document_id=docs[i]['document_id'])
                            for i in order[:depth]]
        return result


def evaluate(path=DATASET, verbose=True):
    dataset = json.loads(Path(path).read_text())
    kb = json.loads((ROOT / dataset['kb_path']).read_text())
    if kb['version'] != dataset['corpus_version']:
        raise ValueError('corpus_version_mismatch')
    cases, docs = dataset['cases'], kb['documents']
    if len(cases) < 24 or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('invalid_benchmark')
    canonical = {(d['tenant_id'], d['document_id']) for d in docs}
    if len(canonical) != len(docs):
        raise ValueError('duplicate_corpus_document')
    client = FixtureRetrievalClient(docs)
    rows = []
    for case in cases:
        tenant = case['tenant_id']
        labels = {doc: 0 for scope, doc in canonical if scope == tenant}
        if set(case['grades']) - set(labels) or not case['rationale']:
            raise ValueError('invalid_labels')
        labels.update(case['grades'])
        rankings = client.retrieve_pair(case, dataset['depth'])
        row = dict(id=case['id'], group=case['group'], rankings={}, scores={})
        for config, hits in rankings.items():
            if any(h['tenant_id'] != tenant or
                   (tenant, h['document_id']) not in canonical for h in hits):
                raise ValueError('cross_tenant_result')
            ids = document_ids(h['document_id'] for h in hits)
            row['rankings'][config] = ids
            row['scores'][config] = score(ids, labels)
        rows.append(row)
    means = {name: mean_scores([r['scores'][name] for r in rows])
             for name in ('dense', 'hybrid')}
    report = dict(dataset_version=dataset['version'], corpus_version=kb['version'],
                  queries=len(rows), depth=dataset['depth'], means=means, rows=rows)
    if verbose:
        print(f"{dataset['version']} / {kb['version']}: {len(rows)} queries; document-level quality")
        print('config       recall@5       MRR    nDCG@5')
        for name, values in means.items():
            print(f"{name:10s} {values['recall@5']:9.4f} {values['rr']:9.4f} {values['nDCG@5']:9.4f}")
        print('Fixture evidence only; offline embeddings are not production embeddings.')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--details', action='store_true')
    args = parser.parse_args()
    report = evaluate()
    if args.details:
        for row in report['rows']:
            print(json.dumps(row, sort_keys=True))


if __name__ == '__main__':
    main()
