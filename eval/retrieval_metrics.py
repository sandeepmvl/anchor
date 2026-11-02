from math import log2


def document_ids(ranked):
    """Preserve first occurrence; inputs are source-document IDs, not chunk IDs."""
    return list(dict.fromkeys(ranked))


def score(ranked, judgments, k=5):
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError('invalid_cutoff')
    if any(not isinstance(g, int) or isinstance(g, bool) or g < 0 or g > 3
           for g in judgments.values()):
        raise ValueError('invalid_grade')
    relevant = {doc for doc, grade in judgments.items() if grade > 0}
    if not relevant:
        raise ValueError('no_relevant_documents')
    ranked = document_ids(ranked)
    if set(ranked) - set(judgments):
        raise ValueError('unjudged_document')
    top = ranked[:k]
    recall = len(set(top) & relevant) / len(relevant)
    rr = next((1 / rank for rank, doc in enumerate(ranked, 1)
               if doc in relevant), 0.0)
    dcg = sum((2 ** judgments[doc] - 1) / log2(rank + 1)
              for rank, doc in enumerate(top, 1))
    ideal = sum((2 ** grade - 1) / log2(rank + 1)
                for rank, grade in enumerate(
                    sorted(judgments.values(), reverse=True)[:k], 1))
    return {'recall@5' if k == 5 else f'recall@{k}': recall,
            'rr': rr, 'nDCG@5' if k == 5 else f'nDCG@{k}': dcg / ideal,
            f'precision@{k}': len(set(top) & relevant) / k}


def mean_scores(rows):
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}
