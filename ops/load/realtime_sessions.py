import importlib.util
import json
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from anchor_ai.realtime import Ledger


def main():
    with tempfile.TemporaryDirectory() as directory:
        ledger = Ledger(Path(directory) / 'surge.sqlite3', sessions=2)
        accepted, rejected = {1: 0, 2: 0}, {1: 0, 2: 0}
        attempts = []
        for tenant in (1, 2):
            for index in range(6):
                scope = {'tenant_id': tenant, 'user_id': 3, 'ticket_id': 11}
                attempts.append((tenant, index, ledger.test_issue(scope)))
        def admit(attempt):
            tenant, index, token = attempt
            try:
                ledger.redeem(token, 'http://testserver', f'{tenant}-{index}')
                return tenant, True
            except OverflowError:
                return tenant, False
        with ThreadPoolExecutor(max_workers=4) as pool:
            for tenant, admitted in pool.map(admit, attempts):
                (accepted if admitted else rejected)[tenant] += 1
        assert accepted == {1: 2, 2: 2}
        assert rejected == {1: 4, 2: 4}
        spec = importlib.util.spec_from_file_location('voice', 'labs/realtime-voice/voice_client.py')
        client = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(client)
        result = client.exercise(directory)
        print(json.dumps({'accepted': accepted, 'tenant_specific_rejections': rejected,
                          'queue_cap': 4, 'fixture_stage_ms': result['stage_ms'],
                          'not_verified': 'distributed capacity, real transport, provider latency'}, indent=2))


if __name__ == '__main__':
    main()
