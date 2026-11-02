import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from anchor_ai.kv_cache import Admission, budget, configuration, metric_view, sample


def fixtures():
    rows = [sample(100, .30, 0, 0, 0),
            sample(101, .93, 8, 12, 100),
            sample(102, .99, 20, 40, 110),
            {'timestamp': 103, 'queued_requests': 21},
            sample(90, .20)]
    previous = None
    for row in rows:
        print(json.dumps(metric_view(row, 104, 5, previous), sort_keys=True))
        previous = row


def simulate(profile='bounded', concurrency=48, config=None):
    started = time.perf_counter()
    if profile not in ('unsafe', 'bounded') or type(concurrency) is not int or not 1 <= concurrency <= 10000:
        raise ValueError('invalid_kv_experiment')
    c = configuration() if config is None else dict(config)
    if profile == 'unsafe':
        # Explicit unsafe candidate: only 100 MB remains before block rounding.
        c.update(gpu_memory_utilization=.70, max_num_seqs=48)
    b = budget(c)
    controller = Admission(c, clock=lambda: 100)
    controller.update(sample(100))
    accepted = []
    for number in range(concurrency):
        if c['max_model_len'] < 512:
            continue
        if profile == 'unsafe' or controller.acquire()['decision'] == 'admitted':
            accepted.append(number)
    resident = min(len(accepted), c['max_num_seqs'])
    capacity = b['sequence_capacity']
    preemptions = 0
    elapsed = 0.0
    completed_at = []
    # All tickets have 480 prompt tokens and 32 output tokens. Full-context
    # reservations are conservative; actual paged allocation grows on demand.
    pending = len(accepted)
    while pending:
        group = min(pending, c['max_num_seqs'])
        for _ in range(32):
            excess = max(0, group - capacity)
            preemptions += excess
            # Assumed iteration cost plus explicit recomputation penalty.
            elapsed += .01 + .002 * group + .25 * excess
        completed_at.extend([elapsed] * group)
        pending -= group
    if profile == 'bounded':
        for _ in accepted:
            controller.release()
    return dict(profile=profile, execution_mode='offline_virtual_time',
                workload='identical_480_prompt_plus_32_output_tokens',
                settings=c, budget=b, offered=concurrency,
                completed=len(accepted), rejected=concurrency - len(accepted),
                peak_resident_sequences=resident,
                queued_requests=max(0, len(accepted) - resident),
                peak_recovery_waiting=max(0, resident - capacity),
                preemptions=preemptions,
                cache_utilization=min(1, resident * b['request_bytes'] / b['pool_bytes']),
                output_tokens_s=len(accepted) * 32 / elapsed if elapsed else 0,
                completed_req_s=len(accepted) / elapsed if elapsed else 0,
                completion_p99_s=completed_at[math.ceil(.99 * len(completed_at)) - 1] if completed_at else None,
                duration_virtual_s=elapsed, retries=0,
                measured_cpu_ms=(time.perf_counter() - started) * 1000,
                production_promotion_allowed=False,
                evidence_warning='No GPU, vLLM inference, swap, CUDA OOM or real throughput verified; recovery costs are assumptions.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixtures', action='store_true')
    parser.add_argument('--profile', choices=['unsafe', 'bounded'], default='bounded')
    parser.add_argument('--block-size', type=int)
    parser.add_argument('--context', type=int)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--utilization', type=float)
    args = parser.parse_args()
    if args.fixtures:
        fixtures()
        return
    c = configuration()
    for field, value in (('block_size', args.block_size), ('max_model_len', args.context),
                         ('max_num_seqs', args.limit), ('gpu_memory_utilization', args.utilization)):
        if value is not None:
            c[field] = value
    print(json.dumps(simulate(args.profile, config=c), sort_keys=True))


if __name__ == '__main__':
    main()
