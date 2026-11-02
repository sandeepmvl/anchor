import argparse
from pathlib import Path
import sys
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from anchor_ai.customization import score


def main():
    parser = argparse.ArgumentParser(description='Anchor customization report; advisory only')
    parser.add_argument('features')
    parser.add_argument('--rare-policy-volatility', type=int)
    args = parser.parse_args()
    data = yaml.safe_load(Path(args.features).read_text(encoding='utf-8'))
    print('Anchor customization report — illustrative evidence; no model activation')
    for candidate in data['features']:
        candidate = dict(candidate)
        if candidate['name'] == 'rare-policy Q&A' and args.rare_policy_volatility is not None:
            candidate['knowledge_volatility'] = args.rare_policy_volatility
        result = score(candidate)
        print(f"\n{result['name']} | policy v{result['policy_version']}")
        print('| Axis | Evidence | Weight | Prompt | RAG | Fine-tune |')
        print('|---|---|---|---|---|---|')
        for axis, row in result['axes'].items():
            points = row['points']
            print(f"| {axis} | {row['value']} | {row['weight']} | "
                  f"{points['prompt']} | {points['RAG']} | {points['fine-tune']} |")
        print('Totals:', result['totals'])
        print('Recommendation:', result['recommendation'])
        print('Evidence:', result['evidence'])
        print('Why:', result['explanation'])
        print('Fine-tune blockers:', ', '.join(result['fine_tune_blockers']) or 'none; experiment only')
    print('\nAdvisory only: refunds not authorized; active model unchanged.')


if __name__ == '__main__':
    main()
