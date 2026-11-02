import argparse
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anchor_ai.dataset_quality import generate, read_rows, write_rows, source_registry
from anchor_ai.customization import load_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', required=True)
    parser.add_argument('--n', required=True, type=int)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    policy = load_policy()
    rows = asyncio.run(generate(read_rows(args.seed), args.n, source_registry(policy), policy))
    write_rows(args.out, rows)
    print(json.dumps({'generated': len(rows), 'provider': rows[0]['generator'],
                      'status': 'sanitized_candidates_not_training_truth'}))


if __name__ == '__main__':
    main()
