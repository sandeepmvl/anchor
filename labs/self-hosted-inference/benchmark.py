import argparse
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from anchor_ai.serving_benchmark import run, compare_artifacts


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', choices=['local', 'hosted'], required=True)
    parser.add_argument('--mode', choices=['offline', 'live'], default='offline')
    parser.add_argument('--concurrency', type=int, choices=[1, 8])
    parser.add_argument('--baseline-artifact')
    parser.add_argument('--candidate-artifact')
    parser.add_argument('--runtime', default='vllm_gpu')
    parser.add_argument('--report')
    args = parser.parse_args()
    if args.baseline_artifact or args.candidate_artifact:
        if not (args.baseline_artifact and args.candidate_artifact) or args.mode != 'offline':
            parser.error('artifact fixture comparison requires both paths and offline mode')
        report = compare_artifacts(args.baseline_artifact, args.candidate_artifact, args.runtime)
        if args.report:
            Path(args.report).write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    for concurrency in ([args.concurrency] if args.concurrency else [1, 8]):
        report = await run(args.target, concurrency, mode=args.mode)
        print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    asyncio.run(main())
