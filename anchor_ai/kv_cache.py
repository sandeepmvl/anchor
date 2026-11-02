"""Full-attention, per-rank planning and bounded application admission."""
import copy
import math
import threading
import time
from pathlib import Path

import yaml

DTYPES = {'float16': 2, 'bfloat16': 2, 'float32': 4, 'fp8': 1}
METRICS = ('timestamp', 'cache_utilization', 'queued_requests',
           'preemptions_total', 'generation_tokens_total')


def positive(value):
    if type(value) is not int or value <= 0:
        raise ValueError('invalid_kv_integer')
    return value


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate(c):
    m = c['model']
    for key in ('layers', 'num_key_value_heads', 'head_dimension'):
        positive(m[key])
    if m['dtype'] not in DTYPES:
        raise ValueError('unsupported_kv_dtype')
    tp = positive(c['tensor_parallel_size'])
    heads = m['num_key_value_heads']
    if max(tp, heads) % min(tp, heads):
        raise ValueError('unsupported_kv_sharding')
    for key in ('vram_bytes_per_rank', 'weights_bytes_per_rank',
                'overhead_bytes_per_rank', 'max_model_len', 'max_num_seqs'):
        positive(c[key])
    if type(c['block_size']) is not int or c['block_size'] not in (16, 32):
        raise ValueError('unsupported_kv_block_size')
    for key in ('gpu_memory_utilization', 'pressure_threshold'):
        if not finite(c[key]) or not 0 < c[key] < 1:
            raise ValueError('invalid_kv_fraction')
    if not finite(c['max_metric_age_s']) or c['max_metric_age_s'] <= 0:
        raise ValueError('invalid_metric_age')
    return c


def configuration(path='config/self_hosted.yml'):
    return validate(yaml.safe_load(Path(path).read_text())['kv_cache'])


def budget(c, sequence_length=None, batch_size=1):
    validate(c)
    length = positive(c['max_model_len'] if sequence_length is None else sequence_length)
    positive(batch_size)
    m = c['model']
    # Equal sharding when heads >= TP; otherwise one replicated head per rank.
    local_heads = max(1, m['num_key_value_heads'] // c['tensor_parallel_size'])
    token = 2 * m['layers'] * local_heads * m['head_dimension'] * DTYPES[m['dtype']]
    block = token * c['block_size']
    blocks = (length + c['block_size'] - 1) // c['block_size']
    available = (int(c['vram_bytes_per_rank'] * c['gpu_memory_utilization'])
                 - c['weights_bytes_per_rank'] - c['overhead_bytes_per_rank'])
    if available < block:
        raise ValueError('no_kv_pool')
    pool_blocks = available // block
    return dict(bytes_per_token=token, local_kv_heads=local_heads,
                useful_bytes=token * length * batch_size,
                request_bytes=blocks * block, batch_bytes=blocks * block * batch_size,
                pool_bytes=pool_blocks * block, pool_blocks=pool_blocks,
                sequence_capacity=pool_blocks // blocks)


def sample(timestamp, utilization=0.1, queued=0, preemptions=0, tokens=0):
    """Explicit normalized fixture; this function does not collect live metrics."""
    return dict(timestamp=timestamp, cache_utilization=utilization,
                queued_requests=queued, preemptions_total=preemptions,
                generation_tokens_total=tokens)


def metric_view(raw, now, max_age, previous=None):
    raw = raw if isinstance(raw, dict) else {}
    missing = []
    for key in METRICS:
        v = raw.get(key)
        valid = finite(v) and v >= 0
        if key == 'cache_utilization':
            valid = valid and v <= 1
        if key in ('queued_requests', 'preemptions_total', 'generation_tokens_total'):
            valid = type(v) is int and v >= 0
        if not valid:
            missing.append(key)
    timestamp = raw.get('timestamp')
    age = now - timestamp if 'timestamp' not in missing else None
    fresh = not missing and age is not None and 0 <= age <= max_age
    result = dict(fresh=fresh, missing=missing, metric_age_s=age,
                  cache_utilization=None if 'cache_utilization' in missing else raw['cache_utilization'],
                  queued_requests=None if 'queued_requests' in missing else raw['queued_requests'],
                  preemptions_delta=None, tokens_per_second=None)
    if fresh and previous:
        prior = metric_view(previous, now, max_age)
        dt = timestamp - previous.get('timestamp', timestamp)
        if prior['fresh'] and dt > 0:
            p = raw['preemptions_total'] - previous['preemptions_total']
            t = raw['generation_tokens_total'] - previous['generation_tokens_total']
            # Counter resets are unknown intervals, not negative rates.
            if p >= 0 and t >= 0:
                result.update(preemptions_delta=p, tokens_per_second=t / dt)
    return result


class Admission:
    """One controller per backend replica; no queue and no retries."""
    def __init__(self, config, clock=time.monotonic):
        self.config = copy.deepcopy(validate(config))
        self.budget = budget(self.config)
        self.clock = clock
        self.active = 0
        self.raw = None
        self.lock = threading.Lock()

    def update(self, raw):
        with self.lock:
            self.raw = copy.deepcopy(raw)

    def acquire(self):
        with self.lock:
            view = metric_view(self.raw, self.clock(), self.config['max_metric_age_s'])
            decision = 'admitted'
            if not view['fresh']:
                decision = 'metrics_unavailable'
            elif view['cache_utilization'] >= self.config['pressure_threshold']:
                decision = 'cache_pressure'
            elif (self.active >= self.config['max_num_seqs'] or
                  (self.active + 1) * self.budget['request_bytes'] > self.budget['pool_bytes']):
                decision = 'capacity'
            if decision == 'admitted':
                self.active += 1
            return dict(decision=decision, active=self.active,
                        reserved_bytes=self.active * self.budget['request_bytes'],
                        fresh=view['fresh'], cache_utilization=view['cache_utilization'])

    def release(self):
        with self.lock:
            if self.active <= 0:
                raise ValueError('kv_reservation_underflow')
            self.active -= 1
