import json
from pathlib import Path
import sys

# A file-based worker must work without an inherited PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from anchor_ai.pii import redact_value

MAX_REQUEST = 262144


def main():
    try:
        raw = sys.stdin.buffer.read(MAX_REQUEST + 1)
        if len(raw) > MAX_REQUEST:
            raise ValueError()
        request = json.loads(raw)
        if set(request) != {'value'}:
            raise ValueError()
        safe = redact_value(request['value'])
        encoded = json.dumps({'value': safe}, ensure_ascii=True, allow_nan=False)
        if len(encoded) > 1048576:
            raise ValueError()
        sys.stdout.write(encoded)
        return 0
    except Exception:
        sys.stderr.write('pii_redaction_unavailable\n')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
