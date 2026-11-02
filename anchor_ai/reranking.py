import math
import os
import re
from typing import Protocol


class PairScorer(Protocol):
    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        ...


class OfflinePairScorer:
    """Term-overlap stand-in, not an attention-based cross-encoder."""

    def score_pairs(self, pairs):
        def terms(text):
            return set(re.findall(r'[a-z0-9]+', text.lower()))
        return [float(len(terms(query) & terms(text))) for query, text in pairs]


class CrossEncoderPairScorer:
    def __init__(self, model_name, batch_size=20, max_length=256):
        # Optional dependency: importing this module does not load model weights.
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, max_length=max_length)
        self.batch_size = batch_size

    def score_pairs(self, pairs):
        return list(self.model.predict(pairs, batch_size=self.batch_size,
                                       show_progress_bar=False))


class Reranker:
    def __init__(self, scorer: PairScorer, candidate_limit=20):
        if type(candidate_limit) is not int or not 1 <= candidate_limit <= 20:
            raise ValueError('invalid_rerank_limit')
        self.scorer = scorer
        self.candidate_limit = candidate_limit

    def rerank(self, query, candidates):
        rows = [dict(row) for row in candidates[:self.candidate_limit]]
        if not rows:
            return []
        scores = list(self.scorer.score_pairs([(query, row['text']) for row in rows]))
        if len(scores) != len(rows):
            raise ValueError('invalid_pair_scores')
        for row, value in zip(rows, scores):
            if isinstance(value, bool):
                raise ValueError('invalid_pair_scores')
            value = float(value)
            if not math.isfinite(value):
                raise ValueError('invalid_pair_scores')
            row['rerank_score'] = value
        return sorted(rows, key=lambda row: -row['rerank_score'])


def make_reranker(settings=None):
    settings = settings or {}
    selected = os.getenv(settings.get('provider_env', 'ANCHOR_RERANKER'), 'disabled')
    if selected == 'disabled':
        return None
    if selected == 'offline':
        scorer = OfflinePairScorer()
    elif selected == 'cross-encoder/ms-marco-MiniLM-L6-v2':
        scorer = CrossEncoderPairScorer(selected,
            batch_size=settings.get('batch_size', 20),
            max_length=settings.get('max_length', 256))
    else:
        raise ValueError('unknown_reranker')
    return Reranker(scorer, settings.get('candidate_limit', 20))
