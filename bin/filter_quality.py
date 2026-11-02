import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anchor_ai.customization import load_policy
from anchor_ai.dataset_quality import read_rows, write_rows, source_registry, filter_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='input', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--rejects', required=True)
    args = parser.parse_args()
    if len({str(Path(p).resolve()) for p in (args.input, args.out, args.rejects)}) != 3:
        raise ValueError('input_and_outputs_must_differ')
    policy = load_policy()
    seed = read_rows(policy['synthetic']['source_paths'][0])
    clean, rejects, summary = filter_rows(read_rows(args.input), seed, source_registry(policy), policy)
    write_rows(args.out, clean)
    write_rows(args.rejects, rejects)
    print(json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    main()
