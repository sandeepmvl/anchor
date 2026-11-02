import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anchor_ai.dataset_quality import read_rows, write_rows
from anchor_ai.preferences import build_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--tenant', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    pairs = build_pairs(read_rows(args.source), args.tenant)
    write_rows(args.out, pairs)
    print(json.dumps({'source': 'constructed_recording_not_measured',
                      'pairs': len(pairs), 'export': args.out,
                      'sample': pairs[0], 'training_release': False}, indent=2))


if __name__ == '__main__':
    main()
