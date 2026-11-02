import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from anchor_ai.adapters import numeric_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixture', required=True, type=Path)
    args = parser.parse_args()
    report = numeric_report(args.fixture)
    assert report['floating_merge_output_error'] < 1e-6
    assert report['wrong_alpha_self_output_error'] < 1e-6
    assert report['wrong_vs_correct_weight_error'] > 0
    assert report['additional_nf4_merge_error'] > 0
    output = Path('.git/anchor-adapter-evidence')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'numeric.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print('Floating fold agrees; wrong-alpha self-check also agrees.')
    print('Quality evidence is separate: run eval/model_gate.py.')


if __name__ == '__main__':
    main()
