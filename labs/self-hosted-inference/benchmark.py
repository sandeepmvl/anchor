import argparse
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from anchor_ai.serving_benchmark import run


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', choices=['local', 'hosted'], required=True)
    parser.add_argument('--mode', choices=['offline', 'live'], default='offline')
    parser.add_argument('--concurrency', type=int, choices=[1, 8])
    args = parser.parse_args()
    for concurrency in ([args.concurrency] if args.concurrency else [1, 8]):
        report = await run(args.target, concurrency, mode=args.mode)
        print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    asyncio.run(main())
