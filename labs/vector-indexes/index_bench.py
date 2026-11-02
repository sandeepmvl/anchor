import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import faiss
import numpy as np
import yaml
from anchor_ai.vector_index import Generation, VectorIndex


def make_data(settings, normalize):
    rng = np.random.default_rng(settings['seed'])
    d = settings['dimensions']
    centers = rng.normal(0, 0.4, (48, d)).astype('float32')
    labels = rng.integers(0, 48, settings['vectors'])
    base = centers[labels] + rng.normal(
        size=(settings['vectors'], d)).astype('float32')
    qrng = np.random.default_rng(settings['seed'] + 1)
    qlabels = qrng.integers(0, 48, settings['queries'])
    queries = centers[qlabels] + qrng.normal(
        size=(settings['queries'], d)).astype('float32')
    if normalize:
        faiss.normalize_L2(base)
        faiss.normalize_L2(queries)
    return base, queries


def report(name, index, queries, reference):
    index.search(queries[:1], 10)
    times, recalls = [], []
    for query, truth in zip(queries, reference):
        start = perf_counter()
        _, ids = index.search(query[None, :], 10)
        times.append(1000 * (perf_counter() - start))
        recalls.append(len(set(ids[0]) & set(truth)) / 10)
    recall = float(np.mean(recalls))
    p50, p99 = np.percentile(times, [50, 99])
    size = faiss.serialize_index(index).nbytes / 1_000_000
    print(f'| {name} | {recall:.4f} | {p50:.4f} | {p99:.4f} | {size:.4f} |')
    return name, round(recall, 8)


def demo():
    generation = Generation('anchor-recorded-embedding-v1', 'authored-fixture-v1', 4)
    def row(tenant, chunk, vector, revision=generation.revision):
        return dict(tenant_id=tenant, chunk_id=chunk, vector=vector,
                    model=generation.model, revision=revision, dimensions=4)
    stored = [row('shop-a', 'refund-policy', [1, 0, 0, 0]),
              row('shop-a', 'shipping-policy', [0, 1, 0, 0]),
              row('shop-b', 'foreign-refund', [1, 0, 0, 0]),
              row('shop-a', 'old-refund', [1, 0, 0, 0], 'retired-v0')]
    index = VectorIndex.from_config(ROOT / 'config/retrieval.yml')
    index.load('shop-a', generation, stored)
    hits = index.search('shop-a', generation, [1, 0, 0, 0], k=2)
    assert [hit['chunk_id'] for hit in hits] == ['refund-policy', 'shipping-policy']
    print('Anchor exact candidates: tenant=shop-a generation=' + generation.revision)
    print(json.dumps(hits, sort_keys=True))
    print('Foreign tenant and retired generation excluded before search.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fingerprint-only', action='store_true')
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()
    faiss.omp_set_num_threads(1)
    if args.demo:
        demo()
        return
    config = yaml.safe_load((ROOT / 'config/retrieval.yml').read_text())['vector_index']
    VectorIndex.from_config(ROOT / 'config/retrieval.yml')
    settings = config['benchmark']
    base, queries = make_data(settings, config['normalize'])
    print(f'Workload: vectors={len(base)} dimensions={base.shape[1]} queries={len(queries)}')
    print('Corpus SHA256:', hashlib.sha256(base.tobytes()).hexdigest())
    print('Query SHA256:', hashlib.sha256(queries.tobytes()).hexdigest())
    if args.fingerprint_only:
        return
    metric = (faiss.METRIC_L2 if config['metric'] == 'l2'
              else faiss.METRIC_INNER_PRODUCT)
    d = base.shape[1]
    flat = faiss.IndexFlat(d, metric)
    flat.add(base)
    reference = flat.search(queries, 10)[1]
    print('Single-thread search only; one warm-up per configuration; decimal MB.')
    print('| configuration | recall@10 | p50 ms | p99 ms | serialized MB |')
    print('|---|---:|---:|---:|---:|')
    signatures = [report('Flat', flat, queries, reference)]
    assert signatures[0][1] == 1.0
    start = perf_counter()
    ivf = faiss.IndexIVFFlat(faiss.IndexFlat(d, metric), d,
                             settings['ivf']['nlist'], metric)
    ivf.cp.seed = settings['seed']
    ivf.cp.niter = settings['ivf']['iterations']
    ivf.train(base)
    ivf.add(base)
    ivf_build = perf_counter() - start
    for probes in settings['ivf']['sweep']:
        ivf.nprobe = probes
        signatures.append(report(f'IVF nprobe={probes}', ivf, queries, reference))
    start = perf_counter()
    h = settings['hnsw']
    graph = faiss.IndexHNSWFlat(d, h['M'], metric)
    graph.hnsw.efConstruction = h['efConstruction']
    graph.add(base)
    graph_build = perf_counter() - start
    for effort in h['sweep']:
        graph.hnsw.efSearch = effort
        signatures.append(report(f'HNSW efSearch={effort}', graph, queries, reference))
    print(f'Build seconds: IVF={ivf_build:.3f}, HNSW={graph_build:.3f}')
    print('Recall signature:', json.dumps(signatures))
    print('Twenty-query p99 is exploratory, not a production tail estimate.')


if __name__ == '__main__':
    main()
