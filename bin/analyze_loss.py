import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anchor_ai.training import analyze_loss, read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.log)
    result = analyze_loss(rows)
    print('Sources:', ', '.join(sorted({r.get('source', 'unverified') for r in rows})))
    print('epoch train_loss val_loss')
    for row in rows:
        print(row['epoch'], row['train_loss'], row['val_loss'])
    print(json.dumps(result, indent=2, sort_keys=True))
    print('Candidate reference only; no weights loaded, created, or promoted.')


if __name__ == '__main__':
    main()
