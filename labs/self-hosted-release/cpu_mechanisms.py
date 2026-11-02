import json
import time
import numpy as np

rng = np.random.default_rng(3752)
x = rng.normal(size=(8, 64)).astype(np.float32)
w = rng.normal(size=(64, 64)).astype(np.float32)
scale = float(np.max(np.abs(w))) / 127
packed = np.clip(np.rint(w / scale), -127, 127).astype(np.int8)
restored = packed.astype(np.float32) * scale
reference = x @ w
quantized = x @ restored


def measure(batched):
    start = time.perf_counter()
    for _ in range(100):
        output = x @ w if batched else np.stack([row @ w for row in x])
    elapsed = time.perf_counter() - start
    assert np.allclose(output, reference, atol=1e-5)
    return {'measured_cpu_ms': elapsed * 1000,
            'request_operations_per_second': 800 / elapsed,
            'mean_iteration_ms': elapsed * 10}


lengths = [17, 31, 64, 95, 128, 160, 256, 300]
blocks = sum((length + 15) // 16 for length in lengths)
kv_bytes = blocks * 16 * 2 * 22 * 4 * 64 * 2
report = dict(individual=measure(False), batch=measure(True),
              kv_block_size=16, kv_allocated_blocks=blocks,
              estimated_kv_bytes=kv_bytes,
              measured_weight_mse=float(np.mean((w - restored) ** 2)),
              measured_output_mse=float(np.mean((reference - quantized) ** 2)),
              limitations='CPU microbenchmark and full-attention KV estimate; not AWQ calibration, model quality, GPU latency or vLLM throughput.')
assert report['measured_weight_mse'] > 0 and kv_bytes > 0
print(json.dumps(report, indent=2))
