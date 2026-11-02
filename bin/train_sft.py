import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anchor_ai.customization import load_policy
from anchor_ai.training import experiment


def main():
    parser = argparse.ArgumentParser(description='Describe an SFT experiment; never launch training.')
    parser.add_argument('--offline', action='store_true', required=True)
    parser.add_argument('--config', default='config/fine_tuning.yml')
    args = parser.parse_args()
    print(json.dumps(experiment(load_policy(args.config)), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
