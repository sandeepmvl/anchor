import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from anchor_ai.hosting_economics import compare


def main():
    parser = argparse.ArgumentParser(description='Offline Anchor hosting planning only')
    parser.add_argument('--usage', default=str(ROOT / 'eval/fixtures/hosting_usage_30_days.jsonl'))
    parser.add_argument('--utilization', type=float, default=0.2)
    parser.add_argument('--operations-usd', type=float, default=180)
    parser.add_argument('--output', default='artifacts/inference_decision/matrix.json')
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / 'config/self_hosted.yml').read_text())['hosting_economics']
    rows = [json.loads(line) for line in Path(args.usage).read_text().splitlines() if line.strip()]
    report = compare(rows, config, args.utilization, args.operations_usd)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + '\n')
    for workload, data in report['workloads'].items():
        print(workload, data['recommendation'], [(r['option'], r['monthly_usd']) for r in data['matrix']])
    print('Production routing disabled. No GPU, provider, quality or live latency verified.')

if __name__ == '__main__':
    main()
