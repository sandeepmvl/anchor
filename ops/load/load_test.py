import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from anchor_ai.serving_benchmark import surge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--concurrency', type=int, default=50)
    parser.add_argument('--backend', choices=['vllm'], default='vllm')
    parser.add_argument('--profile', choices=['default', 'tuned'])
    parser.add_argument('--mode', choices=['continuous', 'static', 'window'], default='continuous')
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--sweep', action='store_true')
    args = parser.parse_args()
    if args.sweep:
        first_miss = None
        for concurrency in (1, 10, 20, 50):
            for mode in ('static', 'window', 'continuous'):
                report = surge(concurrency, args.profile, mode)
                print(json.dumps(report, sort_keys=True))
                if (mode == 'static' and report['short_first_p99_s'] is not None
                        and report['short_first_p99_s'] > .30 and first_miss is None):
                    first_miss = concurrency
        print(json.dumps({'synthetic_short_first_token_objective_s': .30,
                          'first_tested_static_concurrency_missing_objective': first_miss}))
    elif args.compare:
        before = surge(args.concurrency, 'default', args.mode)
        after = surge(args.concurrency, 'tuned', args.mode)
        assert before['workload_sha256'] == after['workload_sha256']
        print(json.dumps({'default': before, 'tuned': after,
                          'completed_throughput_ratio': after['completed_req_s'] / before['completed_req_s'],
                          'p99_change_s': after['latency_p99_s'] - before['latency_p99_s']}, sort_keys=True))
    else:
        print(json.dumps(surge(args.concurrency, args.profile, args.mode), sort_keys=True))


if __name__ == '__main__':
    main()
