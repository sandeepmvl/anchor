import re
from pathlib import Path

import yaml
from rank_bm25 import BM25Okapi

from .index_maintenance import retrieval_snapshot, valid_tenant
from .schemas import SparseHit


def tokenize(text):
    # Keep numeric identifiers and internal code punctuation intact.
    return re.findall(r'\w+(?:[-._]\w+)*', text.lower())


class SparseRetriever:
    def __init__(self, db, top_k=None):
        self.db = db
        path = Path(__file__).resolve().parents[1] / 'config/retrieval.yml'
        settings = yaml.safe_load(path.read_text())['sparse']
        if settings['tokenizer_version'] != 'codes-v1':
            raise ValueError('unknown_sparse_tokenizer')
        self.k1, self.b = float(settings['k1']), float(settings['b'])
        self.top_k = settings['top_k'] if top_k is None else top_k
        if not self.k1 > 0 or not 0 <= self.b <= 1:
            raise ValueError('invalid_bm25_settings')
        if type(self.top_k) is not int or self.top_k < 1:
            raise ValueError('invalid_top_k')
        # Keep only the latest snapshot for each resident tenant.
        self.indexes = {}

    def search(self, query, tenant_id):
        valid_tenant(tenant_id)
        terms = tokenize(query)
        if not terms:
            return []
        gen, rows = retrieval_snapshot(self.db, tenant_id)
        if gen is None:
            self.indexes.pop(tenant_id, None)
            return []
        key = (gen['id'], gen['epoch'])
        cached = self.indexes.get(tenant_id)
        if cached is None or cached[0] != key:
            tokens = [tokenize(row['text']) for row in rows]
            index = BM25Okapi(tokens, k1=self.k1, b=self.b) if any(tokens) else None
            cached = (key, rows, tokens, index)
            self.indexes[tenant_id] = cached
        _, rows, tokens, index = cached
        if index is None:
            return []
        scores = index.get_scores(terms)
        query_terms = set(terms)
        hits = [
            SparseHit(id=row['id'], tenant_id=tenant_id, text=row['text'],
                      score=float(score)).model_dump()
            for row, words, score in zip(rows, tokens, scores)
            if query_terms.intersection(words)
        ]
        # Overlap, not score > 0, defines candidates: Okapi scores can be negative.
        return sorted(hits, key=lambda hit: (-hit['score'], hit['id']))[:self.top_k]
