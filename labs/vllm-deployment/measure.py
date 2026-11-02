import asyncio
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from anchor_ai.llm_client import LlmClient
from anchor_ai.provider import ReplayProvider
from anchor_ai.vllm_backend import VllmBackend, validate_deployment


def measure():
    rng = np.random.default_rng(3748)
    weights = rng.normal(0, 0.1, (64, 64)).astype(np.float32)
    inputs = rng.normal(size=(8, 64)).astype(np.float32)
    scale = max(float(np.max(np.abs(weights))) / 7, 1e-12)
    quantized = np.clip(np.round(weights / scale), -7, 7) * scale
    baseline = inputs @ weights
    approximation = inputs @ quantized
    repeats = 100
    start = time.perf_counter()
    for _ in range(repeats):
        single = np.concatenate([row[None, :] @ weights for row in inputs])
    single_seconds = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(repeats):
        batched = inputs @ weights
    batch_seconds = time.perf_counter() - start
    np.testing.assert_allclose(single, batched, atol=1e-5)
    # K and V, layers, KV heads, head dimension, FP16 bytes, tokens.
    per_token = 2 * 22 * 4 * 64 * 2
    token_lengths = [17, 33, 65, 127, 129, 257, 300, 511]
    block = 16
    allocated_tokens = sum(((n + block - 1) // block) * block for n in token_lengths)
    return {'basis': 'measured_cpu_numpy_not_vllm_or_awq',
            'four_bit_rounding_output_rmse': float(np.sqrt(np.mean((baseline - approximation) ** 2))),
            'batch_max_absolute_difference': float(np.max(np.abs(single - batched))),
            'single_row_ms': single_seconds * 1000 / (8 * repeats),
            'batch_ms': batch_seconds * 1000 / repeats,
            'batch_rows_per_second': 8 * repeats / batch_seconds,
            'kv_basis': 'calculated_fp16_cache_not_measured_gpu_memory',
            'kv_bytes_per_token': per_token,
            'kv_live_bytes': sum(token_lengths) * per_token,
            'kv_paged_bytes': allocated_tokens * per_token,
            'kv_contiguous_reserved_bytes': 8 * 512 * per_token}


async def compare():
    rows = []
    for name, provider in [('hosted_adapter_offline', ReplayProvider()),
                           ('self_hosted_adapter_offline', VllmBackend())]:
        start = time.perf_counter()
        result = await LlmClient(provider).complete_result(
            'Evidence: Refunds take five days. [kb:refund-policy]\nWhen will my refund arrive?')
        elapsed = (time.perf_counter() - start) * 1000
        assert result.finish_reason == 'stop'
        assert result.usage.total_tokens == result.usage.prompt_tokens + result.usage.completion_tokens
        rows.append({'backend': name, 'elapsed_ms': elapsed,
                     'completed_contract': True, 'tokens': result.usage.total_tokens,
                     'contains_refund_timing': 'five' in result.reply.lower(),
                     'basis': 'offline_protocol_fixture_not_model_quality_or_live_latency'})
    return rows


if __name__ == '__main__':
    spec = validate_deployment()
    print(json.dumps({'deployment_configuration_valid': True,
                      'runtime': spec['runtime'], 'gpu_deployment_verified': False,
                      'mechanism': measure(), 'single_requests': asyncio.run(compare()),
                      'release': 'refused: real infrastructure evidence missing'}, indent=2))
