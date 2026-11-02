import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anchor_ai.pii import redact_finetuning_batch


def main():
    try:
        rows = json.load(sys.stdin)
        if not isinstance(rows, list):
            raise ValueError()
        safe = redact_finetuning_batch(rows)
        encoded = json.dumps(safe)
    except Exception:
        print('finetuning_redaction_failed', file=sys.stderr)
        return 1
    print(encoded)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
