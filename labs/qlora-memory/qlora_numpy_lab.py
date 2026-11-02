import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from anchor_ai.training import estimate_training_memory

GB = 1_000_000_000
LORA = {
    'r': 64,
    'lora_alpha': 128,
    'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                       'gate_proj', 'up_proj', 'down_proj'],
}
BNB_PLAN = {
    'load_in_4bit': True,
    'bnb_4bit_quant_type': 'nf4',
    'bnb_4bit_use_double_quant': True,
    'bnb_4bit_compute_dtype': 'bfloat16 on supported Ampere+; otherwise validate float32',
}


def budget(parameters, hidden, intermediate, layers, seq, mode, checkpoint, card=16):
    # Illustrative square attention projections, not Anchor's Qwen inventory.
    adapter = layers * LORA['r'] * (4 * (2 * hidden) + 3 * (hidden + intermediate))
    # Batch 1, two-byte saved tensors, memory-efficient attention assumed.
    saved_all = seq * layers * hidden * 16 * 2
    boundaries = seq * layers * hidden * 2
    workspace = seq * (16 * hidden + 3 * intermediate) * 2
    activations = boundaries + workspace if checkpoint else saved_all
    return estimate_training_memory(parameters, adapter, mode, activations, GB, card * GB)


def print_budget():
    print('PLANNING ESTIMATES, decimal GB; NOT completed training')
    print('Inline LoRA:', json.dumps(LORA))
    print('bitsandbytes plan:', json.dumps(BNB_PLAN))
    print('7B: hidden=4096, intermediate=11008, layers=32, sequence=2048, batch=1')
    print('Two-byte adapters/gradients/activations; two fp32 AdamW moments')
    for mode in ('fp16', '8-bit', 'NF4'):
        for checkpoint in (False, True):
            report = budget(7_000_000_000, 4096, 11008, 32, 2048, mode, checkpoint)
            terms = {k: round(v / GB, 3) for k, v in report['components_bytes'].items()}
            total = report['estimated_total_bytes'] / GB
            print(mode, 'checkpoint=', checkpoint, json.dumps(terms),
                  f'total={total:.3f} GB',
                  '16GB=' + ('FIT' if total <= 16 else 'NO_FIT'),
                  '24GB=' + ('FIT' if total <= 24 else 'NO_FIT'))
    large = budget(65_000_000_000, 8192, 22016, 80, 512, 'NF4', True, 48)
    total = large['estimated_total_bytes'] / GB
    print('65B coarse sanity: batch=1, sequence=512, checkpointing=True, rank=64')
    print(json.dumps(large['components_bytes'], sort_keys=True))
    print(f'65B estimate={total:.3f} GB; paper reference=48 GB; NOT a reproduction')
    assert abs(total - 48) / 48 <= 0.20
    assert not budget(7_000_000_000, 4096, 11008, 32, 2048, 'fp16', True)['accepted']
    assert budget(7_000_000_000, 4096, 11008, 32, 2048, 'NF4', True)['accepted']


def nf4_example():
    # Rounded NF4 codebook. One synthetic block; indices are unpacked uint8 here.
    codebook = np.array([
        -1.0, -0.6961928, -0.5250731, -0.3949175,
        -0.2844414, -0.1847734, -0.0910500, 0.0,
        0.0795803, 0.1609302, 0.2461123, 0.3379152,
        0.4407098, 0.5626170, 0.7229568, 1.0,
    ], dtype=np.float32)
    weights = np.random.default_rng(3686).normal(size=64).astype(np.float32)
    scale = np.max(np.abs(weights))
    normalized = weights / scale
    codes = np.argmin(np.abs(normalized[:, None] - codebook), axis=1).astype(np.uint8)
    reconstructed = codebook[codes] * scale
    error = float(np.mean((weights - reconstructed) ** 2))
    print(f'Illustrative NF4 quantize/dequantize: reconstruction MSE={error:.8f}')
    print('Indices 0..15 represent four-bit codes; this demo does not pack them.')
    assert 0 < error < float(np.mean(weights ** 2))
    assert not np.array_equal(weights, reconstructed)


def loss_reduction(accumulation_dtype):
    losses = []
    print('compute=float32; accumulation=', np.dtype(accumulation_dtype))
    for step in range(1, 11):
        per_sample = np.full(8, 4.0 - step / 10, dtype=np.float32)
        weights = np.full(8, 100 if step < 3 else 10_000, dtype=np.float32)
        with np.errstate(over='ignore', invalid='ignore'):
            numerator = np.sum(weights * per_sample, dtype=accumulation_dtype)
            denominator = np.sum(weights, dtype=accumulation_dtype)
            loss = numerator / denominator
        print(f'step={step:02d} numerator={numerator} denominator={denominator} loss={loss}')
        losses.append(float(loss))
    return np.array(losses)


def main():
    print_budget()
    nf4_example()
    print('Constructed numerical repro, NOT a model training curve')
    broken = loss_reduction(np.float16)
    fixed = loss_reduction(np.float32)
    assert np.isfinite(broken[:2]).all() and np.isnan(broken[2:]).all()
    assert np.isfinite(fixed).all()
    print('PASS: fp16 reduction becomes NaN; all ten fp32 losses remain finite.')
    print('Use supported bf16 or fp32 paths; loss scaling cannot repair forward inf/inf.')


if __name__ == '__main__':
    main()
