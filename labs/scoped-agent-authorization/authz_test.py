import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ, ANCHOR_AUTHZ_OFFLINE='1', ANCHOR_LLM_PROVIDER='fake')
    subprocess.run(['ruby', 'test/integration/tool_authorization_test.rb'],
                   cwd=root, env=environment, check=True)
    subprocess.run([sys.executable, '-m', 'pytest', '-q',
                    'anchor_ai/tests/test_tools.py', '-k', 'rails_backend'],
                   cwd=root, env=environment, check=True)
    print('PASS Anchor: approved local refund reaches executor; foreign orders, forged scope,', flush=True)
    print('insufficient roles, and missing or expired authorization stop before execution.', flush=True)
    print('NOT VERIFIED: browser login, production cookie/CSRF transport, payment provider, deployment routing.', flush=True)


if __name__ == '__main__':
    main()
