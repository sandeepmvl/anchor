"""Deterministic lexical judge fake; not semantic quality approval."""
import json
import re
from pathlib import Path

from eval.faithfulness import heuristic_support
from eval.triad import DIMENSIONS, METRICS, fingerprint, normalize, validate_score

METRIC_NAMES = (*DIMENSIONS, 'citation_support')


def terms(text):
    if not isinstance(text, str):
        raise ValueError('invalid_text')
    return set(re.findall(r'[a-z0-9]+', text.lower()))


def overlap(question, text):
    required = terms(question)
    if not required:
        raise ValueError('empty_question')
    return len(required & terms(text)) / len(required)


def score_row(row):
    support = heuristic_support(row)
    contexts = ' '.join(chunk['text'] for chunk in row['evidence'])
    scores = dict(context_relevance=overlap(row['question'], contexts),
                  groundedness=support['groundedness'],
                  answer_relevance=overlap(row['question'], row['answer']),
                  citation_support=support['citation_support'])
    return {name: validate_score(scores[name]) for name in METRIC_NAMES}


def score_capture(captured):
    provenance = captured['provenance']
    required = {'run_id', 'commit_sha', 'config_hash', 'dataset_hash', 'flag'}
    if set(provenance) != required:
        raise ValueError('invalid_provenance')
    if any(not isinstance(provenance[key], str) or not provenance[key]
           for key in required - {'flag'}):
        raise ValueError('missing_provenance')
    if type(provenance['flag']) is not bool:
        raise ValueError('invalid_flag')
    rows = captured['rows']
    ids = [row['id'] for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError('empty_batch_or_duplicate_id')
    results = []
    for row in sorted(rows, key=lambda item: item['id']):
        scores = score_row(row)
        adapters = {}
        for framework, mapping in METRICS.items():
            raw = {name: scores[dimension] for name, dimension in mapping.items()}
            adapters[framework] = dict(raw_scores=raw,
                                       scores=normalize(framework, raw))
        results.append(dict(id=row['id'], tenant_id=row['tenant_id'],
            generation_id=row['generation_id'], input_fingerprint=fingerprint(row),
            scores=scores, adapters=adapters))
    return dict(schema_version=1, judge='heuristic-term-overlap-v1',
        mode='fresh-offline', provenance=dict(provenance),
        capture_hash=fingerprint(captured), rows=results,
        aggregates={name: sum(row['scores'][name] for row in results) / len(results)
                    for name in METRIC_NAMES})


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_bytes(json_bytes(value))
    temporary.replace(path)
