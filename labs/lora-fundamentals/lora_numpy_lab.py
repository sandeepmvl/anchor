import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from anchor_ai.training import digest, lora_forward


def factors(n, rank, singular_values):
    a = np.zeros((rank, n), dtype=np.float64)
    b = np.zeros((n, rank), dtype=np.float64)
    indices = np.arange(rank)
    roots = np.sqrt(singular_values[:rank])
    a[indices, indices] = roots
    b[indices, indices] = roots
    return b, a


def main():
    parser = argparse.ArgumentParser(description='Anchor adapter geometry, not training')
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=16)
    args = parser.parse_args()
    if not 1 <= args.rank <= 64 or not np.isfinite(args.alpha) or args.alpha <= 0:
        parser.error('rank must be 1..64 and alpha must be finite and positive')
    n = 4096
    # Fixed recorded teaching target, represented by its nonzero diagonal.
    singular_values = np.linspace(8.0, 0.25, 64)
    target_diagonal = np.zeros(n)
    target_diagonal[:64] = singular_values
    target_digest = digest(target_diagonal.tolist())
    target_norm = float(np.linalg.norm(singular_values))
    percent = 100 * args.rank * (n + n) / (n * n)
    print(f'W=({n}, {n}); A=({args.rank}, {n}); B=({n}, {args.rank})')
    print(f'trainable_parameter_percent={percent:.6f}; alpha/r={args.alpha / args.rank:g}')
    print(f'fixed_target_digest={target_digest}')
    print('rank parameters percent alpha/r unscaled_error_F relative_error')
    rows = []
    for rank in sorted({1, 2, 4, 8, 16, 32, 64, args.rank}):
        b, a = factors(n, rank, singular_values)
        reconstructed = np.einsum('ir,ri->i', b, a)
        error = float(np.linalg.norm(target_diagonal - reconstructed))
        row = {'rank': rank, 'parameters': int(a.size + b.size),
               'percent': 100 * (a.size + b.size) / (n * n),
               'scale': args.alpha / rank, 'unscaled_error_F': error,
               'relative_error': error / target_norm}
        rows.append(row)
        print(f"{rank:4d} {row['parameters']:10d} {row['percent']:.6f} "
              f"{row['scale']:.3f} {error:.6f} {row['relative_error']:.6f}")
    assert all(a['unscaled_error_F'] > b['unscaled_error_F']
               for a, b in zip(rows, rows[1:]))
    if args.rank == 8:
        assert percent < 1
    # Small projection verifies the same shared forward rule used by tests.
    w = np.eye(4)
    before = w.copy()
    a = np.arange(8, dtype=float).reshape(2, 4) / 10
    b = np.ones((4, 2))
    x = np.linspace(-1, 1, 4)
    output = lora_forward(w, a, b, x, args.alpha)
    merged = (w + (args.alpha / 2) * (b @ a)) @ x
    assert np.allclose(output, merged)
    assert np.array_equal(w, before)
    report = {'shape': [n, n], 'selected_rank': args.rank, 'alpha': args.alpha,
              'trainable_parameter_percent': percent, 'target_digest': target_digest,
              'evidence_status': 'constructed_geometry_not_language_quality',
              'merge_equivalent': True, 'base_unchanged': True, 'rows': rows}
    destination = Path('artifacts/lora_rank_sweep.json')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'Merge equivalence and frozen base checked; report: {destination}')


if __name__ == '__main__':
    main()
