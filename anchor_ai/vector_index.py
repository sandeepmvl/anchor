from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class Generation:
    model: str
    revision: str
    dimensions: int

    def __post_init__(self):
        if not self.model or not self.revision or self.dimensions < 1:
            raise ValueError('invalid embedding generation')


class VectorIndex:
    """In-memory exact baseline; callers supply an authorized tenant ID.

    Rows contain tenant_id, chunk_id, model, revision, dimensions, vector.
    Each load replaces one complete partition, not the entire index.
    """

    def __init__(self, metric='cosine', normalize=True):
        if metric not in {'cosine', 'inner_product', 'l2'}:
            raise ValueError('unsupported metric')
        if metric == 'cosine' and not normalize:
            raise ValueError('cosine requires normalization')
        self.metric = metric
        self.normalize = normalize
        self._partitions = {}

    @classmethod
    def from_config(cls, path='config/retrieval.yml'):
        settings = yaml.safe_load(Path(path).read_text())['vector_index']
        if settings['index_type'] != 'flat':
            raise ValueError('product interface currently supports flat only')
        return cls(settings['metric'], settings['normalize'])

    def _prepare(self, vectors, dimensions):
        array = np.array(vectors, dtype=np.float32, copy=True)
        if array.ndim != 2 or array.shape[1] != dimensions:
            raise ValueError('embedding dimension mismatch')
        if not np.isfinite(array).all():
            raise ValueError('vectors must be finite')
        if self.normalize:
            norms = np.linalg.norm(array, axis=1, keepdims=True)
            if np.any(norms == 0):
                raise ValueError('cannot normalize zero vectors')
            array /= norms
        return array

    @staticmethod
    def _key(tenant_id, generation):
        if not isinstance(tenant_id, str) or not tenant_id:
            raise ValueError('tenant_id must be a nonempty string')
        return tenant_id, generation

    def load(self, tenant_id, generation, rows):
        key = self._key(tenant_id, generation)
        selected = [row for row in rows
                    if row['tenant_id'] == tenant_id
                    and row['model'] == generation.model
                    and row['revision'] == generation.revision]
        if any(row['dimensions'] != generation.dimensions for row in selected):
            raise ValueError('embedding dimension mismatch')
        ids = [str(row['chunk_id']) for row in selected]
        if len(set(ids)) != len(ids):
            raise ValueError('duplicate chunk IDs within partition')
        vectors = ([row['vector'] for row in selected] if selected
                   else np.empty((0, generation.dimensions), dtype=np.float32))
        prepared = self._prepare(vectors, generation.dimensions)
        self._partitions[key] = (np.array(ids, dtype=str), prepared)
        return len(ids)

    def search(self, tenant_id, generation, query, k=10):
        key = self._key(tenant_id, generation)
        if k < 1:
            raise ValueError('k must be positive')
        q = self._prepare([query], generation.dimensions)[0]
        if key not in self._partitions:
            return []
        ids, vectors = self._partitions[key]
        if self.metric == 'l2':
            scores = np.sum((vectors - q) ** 2, axis=1)
            order = np.lexsort((ids, scores))
        else:
            scores = vectors @ q
            order = np.lexsort((ids, -scores))
        return [{'chunk_id': str(ids[i]), 'score': float(scores[i])}
                for i in order[:k]]
