import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anchor_ai.customization import sft_activation_decision
from anchor_ai.training import experiment, recorded_evaluation


def replay(candidate):
    manifest = experiment()
    if candidate != manifest['candidate']:
        raise ValueError('unknown_candidate')
    evidence = recorded_evaluation(manifest)
    decision = sft_activation_decision(manifest, evidence)
    return {'evaluation': evidence, 'decision': decision,
            'retrieval_evidence': 'not_run_for_a_trained_candidate',
            'deployment_approved': decision['active']}


def main():
    parser = argparse.ArgumentParser(description='Replay recorded evidence, not a deployment gate pass.')
    parser.add_argument('--offline', action='store_true', required=True)
    parser.add_argument('--candidate', required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.candidate), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
