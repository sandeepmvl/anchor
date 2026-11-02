from dataclasses import dataclass, field
from hashlib import sha256
import json
import os
import re
import numpy as np


@dataclass
class Node:
    id: str
    text: str
    start: int
    end: int
    metadata: dict = field(default_factory=dict)

    @property
    def embedding_text(self):
        prefix = self.metadata.get('context_prefix', '')
        return prefix + ('\n' if prefix else '') + self.text


def embed_sentences(texts):
    provider = os.getenv('ANCHOR_CHUNK_EMBED_PROVIDER', 'offline')
    if provider != 'offline':
        from anchor_ai.llm_client import LlmClient
        return LlmClient(provider=provider).embed(texts)
    vectors = []
    for text in texts:
        vector = [0.0] * 64
        for word in re.findall(r'\w+', text.lower()):
            digest = sha256(word.encode()).digest()
            vector[int.from_bytes(digest[:2], 'big') % 64] += 1.0
        if not any(vector):
            vector[0] = 1.0
        vectors.append(vector)
    return vectors


def sentence_spans(text):
    # Fixture tokenizer: production ingestion needs abbreviation-aware segmentation.
    spans = []
    for match in re.finditer(r'[^.!?]+(?:[.!?]+|$)', text):
        a, b = match.span()
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        if a < b:
            spans.append((a, b))
    return spans


def node(text, start, end, **metadata):
    return Node('', text[start:end], start, end, metadata)


class FixedSizeChunker:
    def __init__(self, size=100, overlap=20):
        if not isinstance(size, int) or not isinstance(overlap, int) or not 0 <= overlap < size:
            raise ValueError('Require integer 0 <= overlap < size')
        self.size, self.overlap = size, overlap

    def end_at(self, text, start):
        return min(start + self.size, len(text))

    def split(self, text):
        nodes, start, previous_end = [], 0, 0
        while start < len(text):
            end = self.end_at(text, start)
            nodes.append(node(text, start, end, overlap=max(0, previous_end - start)))
            if end == len(text):
                break
            previous_end, start = end, end - self.overlap
        return nodes


class RecursiveChunker(FixedSizeChunker):
    def end_at(self, text, start):
        cap = min(start + self.size, len(text))
        if cap == len(text):
            return cap
        for separator in (r'\n\s*\n', r'[.!?](?:\s+|$)', r'\s+'):
            ends = [m.end() for m in re.finditer(separator, text[start:cap])
                    if m.end() > self.overlap]
            if ends:
                return start + ends[-1]
        return cap


class SemanticChunker:
    def __init__(self, embed_fn=embed_sentences, percentile=75):
        if not callable(embed_fn) or not 0 <= percentile <= 100:
            raise ValueError('Require an embedding callable and percentile in [0, 100]')
        self.embed_fn, self.percentile = embed_fn, percentile
        self.boundaries = []

    def split(self, text):
        spans = sentence_spans(text)
        self.boundaries = []
        if not spans:
            return []
        vectors = np.asarray(self.embed_fn([text[a:b] for a, b in spans]), dtype=float)
        if (vectors.ndim != 2 or vectors.shape[0] != len(spans)
                or vectors.shape[1] == 0 or not np.isfinite(vectors).all()):
            raise ValueError('Require one finite numeric vector per sentence')
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(norms == 0):
            raise ValueError('Cosine similarity requires nonzero vectors')
        unit = vectors / norms[:, None]
        distances = 1 - np.clip(np.sum(unit[:-1] * unit[1:], axis=1), -1, 1)
        threshold = float(np.percentile(distances, self.percentile)) if len(distances) else None
        nodes, start = [], 0
        for i, distance in enumerate(distances):
            record = dict(offset=spans[i + 1][0], breakpoint_score=float(distance),
                          similarity=float(1 - distance), threshold=threshold,
                          split=bool(distance > threshold))
            self.boundaries.append(record)
            if record['split']:
                nodes.append(node(text, start, record['offset'], **record))
                start = record['offset']
        nodes.append(node(text, start, len(text), threshold=threshold))
        return nodes


class SentenceWindowChunker:
    def __init__(self, window=1):
        if not isinstance(window, int) or window < 0:
            raise ValueError('Window must be a nonnegative integer')
        self.window = window

    def split(self, text):
        spans = sentence_spans(text)
        nodes = []
        for i, (a, b) in enumerate(spans):
            first, last = max(0, i - self.window), min(len(spans), i + self.window + 1)
            nodes.append(node(text, a, b,
                              window_before=[text[x:y] for x, y in spans[first:i]],
                              window_after=[text[x:y] for x, y in spans[i + 1:last]],
                              window_start=spans[first][0], window_end=spans[last - 1][1]))
        return nodes


class HierarchicalChunker:
    def __init__(self, parent_size=400, child_size=80, child_overlap=10):
        if child_size > parent_size:
            raise ValueError('Children must fit inside parents')
        self.parents = RecursiveChunker(parent_size, 0)
        self.children = RecursiveChunker(child_size, child_overlap)

    def split(self, text):
        nodes = []
        for index, parent in enumerate(self.parents.split(text)):
            parent.id = f'parent-{index}'
            parent.metadata.update(level='parent', indexable=False)
            nodes.append(parent)
            for child in self.children.split(parent.text):
                child.start += parent.start
                child.end += parent.start
                child.metadata.update(level='child', parent_id=parent.id, indexable=True)
                nodes.append(child)
        return nodes


class StructuralChunker:
    def __init__(self, document, max_chars=4000):
        self.document, self.max_chars = document, max_chars

    def split(self, text):
        if any(u.end - u.start > self.max_chars for u in self.document.units):
            raise ValueError('Structural unit exceeds cap; apply deliberate secondary splitting')
        return [node(text, u.start, u.end, **u.metadata) for u in self.document.units]


def chunk_document(document, strategy, context=False, *, embed_fn=None, options=None):
    options = dict(options or {})
    factories = {
        'fixed_size': FixedSizeChunker,
        'recursive': RecursiveChunker,
        'semantic': lambda **kw: SemanticChunker(embed_fn or embed_sentences, **kw),
        'sentence_window': SentenceWindowChunker,
        'hierarchical': HierarchicalChunker,
        'structural': lambda **kw: StructuralChunker(document, **kw),
    }
    if strategy not in factories:
        raise ValueError(f'Unknown strategy: {strategy}')
    chunker = factories[strategy](**options)
    nodes = chunker.split(document.text)
    # Include algorithm version and settings so distinct boundary policies do not collide.
    identity = json.dumps([document.source, document.version, 'chunking-v1', strategy,
                           options], sort_keys=True)
    old_parents = {}
    for item in nodes:
        old_id = item.id
        key = json.dumps([identity, item.start, item.end, item.metadata.get('level')])
        item.id = sha256(key.encode()).hexdigest()
        if old_id:
            old_parents[old_id] = item.id
    for item in nodes:
        if 'parent_id' in item.metadata:
            item.metadata['parent_id'] = old_parents[item.metadata['parent_id']]
        unit = next((u for u in document.units if u.start <= item.start < u.end), None)
        if unit:
            for key, value in unit.metadata.items():
                item.metadata.setdefault(key, value)
        item.metadata.update(source=document.source, source_version=document.version,
                             strategy=strategy)
        item.metadata.setdefault('indexable', True)
        if context:
            label = (item.metadata.get('section') or item.metadata.get('question')
                     or item.metadata.get('speaker') or document.title)
            item.metadata['context_prefix'] = (
                f'Source: {document.source}; title: {document.title}; section: {label}')
    return nodes, getattr(chunker, 'boundaries', [])
