import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anchor_ai.browser_tools import FakeDriver, PlaywrightDriver, run_agent
from anchor_ai.sandbox import SandboxService


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixture', choices=['clean', 'injected', 'account'], default='clean')
    parser.add_argument('--readiness', action='store_true')
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--network-denied', action='store_true')
    args = parser.parse_args()
    if args.live:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            # Use Firefox already installed in the supplied Playwright image.
            browser_type = p.firefox
            executable = Path(browser_type.executable_path)
            if not executable.is_file():
                raise RuntimeError('preinstalled_browser_missing; do not download at runtime')
            print(json.dumps({'browser_executable_present': True, 'downloads_required': False}))
            with browser_type.launch(headless=True) as browser:
                if args.readiness:
                    print(json.dumps({'browser_launched': True, 'version': browser.version}))
                    return
                handler = partial(SimpleHTTPRequestHandler, directory=str(ROOT / 'labs/browser-agents'))
                server = ThreadingHTTPServer(('127.0.0.1', 8765), handler)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    result = run_agent(PlaywrightDriver(browser, args.fixture))
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=2)
    elif args.readiness:
        print(json.dumps(SandboxService(rehearsal=True).browser_readiness()))
        return
    elif args.network_denied:
        result = run_agent(FakeDriver(network_denied=True))
    else:
        wrapped = SandboxService(rehearsal=True).execute_fixture(1, 'browser_' + args.fixture)
        if wrapped['exit_reason'] != 'exited' or wrapped['returncode'] != 0:
            raise RuntimeError('browser_worker_failed: ' + wrapped['exit_reason'])
        result = json.loads(wrapped['output'])
    for event in result['trace']:
        print(json.dumps(event, sort_keys=True))
    print('Anchor ticket A100:', result['status'] or result['outcome'])
    expected = 'network_error' if args.network_denied else {
        'clean': 'result', 'injected': 'blocked', 'account': 'approval_required'}[args.fixture]
    assert result['outcome'] == expected
    if expected == 'result':
        assert result['status'] == 'Delivered'


if __name__ == '__main__':
    main()
