import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anchor_ai.dataset_quality import read_rows
from anchor_ai.preferences import analyze
from anchor_ai.training import dpo_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs', required=True)
    parser.add_argument('--replies', required=True)
    args = parser.parse_args()
    manifest = dpo_experiment()
    report = analyze(read_rows(args.logs), read_rows(args.replies), manifest['heldout_ids'])
    print('CONSTRUCTED OFFLINE RECORDINGS — not measured GPU training')
    print('HELD-OUT PREFERENCE PAIRS')
    for row in report['preference']:
        print(f"step={row['step']} N={row['n']} win_rate={row['win_rate']:.1%} "
              f"margin={row['mean_margin']:.3f} loss={row['loss']:.4f} ties={row['ties']}")
    print('RESOLVED-TICKET EVAL — independent rubric; handoff is not a failed action')
    for stage, row in report['resolved_ticket_eval'].items():
        print(f"{stage}: N={row['n']} resolved={row['resolved_rate']:.0%} "
              f"appropriate={row['appropriate_rate']:.0%} generic={row['generic_rate']:.0%}")
    if report['reward_hacking_flag']:
        print('BLOCK: preference proxy rises while independent product quality falls.')
        print('Inspect replies and rejected coverage; compare earlier stopping, higher beta,')
        print('reviewed generic negatives, and length-aware evaluation. Rerun all gates.')
    print(json.dumps({'metrics_scope': manifest['metrics_scope'],
                      'training_launched': False, 'deployment_approved': False}))


if __name__ == '__main__':
    main()
