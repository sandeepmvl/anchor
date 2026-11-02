"""Run from Anchor's root; no server, credentials, or network required."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    env = dict(os.environ, ANCHOR_LLM_PROVIDER='offline')
    command = [sys.executable, '-m', 'pytest', '-q',
               'anchor_ai/tests/test_injection_filter.py',
               'anchor_ai/tests/test_tools.py',
               '-k', 'injection or untrusted_evidence or flagged_customer or normalization or '
                     'rag_generation_route or only_flagged or paraphrase or missing_service']
    subprocess.run(command, cwd=ROOT, env=env, check=True, timeout=9)
    print('Anchor: hidden refund instruction quarantined; safe policy remains usable; '
          'no trusted approval supplied; refund dispatches = 0.')
    print('Offline scope: existing Python HTTP routes and dispatcher assertions. '
          'Not verified: live-model resistance, Rails authentication, production retrieval '
          'tenant isolation, payment-provider behavior, or diagnostic-sink permissions.')


if __name__ == '__main__':
    main()
