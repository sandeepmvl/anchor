import asyncio
import contextlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'labs/code-sandboxes/malicious_script.py'


class SandboxArguments(BaseModel):
    model_config = ConfigDict(extra='forbid')
    script_id: str = 'ticket_diagnostic'
    confirmed: bool = False
    dry_run: bool = True


class SandboxResult(BaseModel):
    model_config = ConfigDict(extra='forbid')
    exit_reason: str
    output: str = ''
    returncode: int | None = None
    backend: str = 'rehearsal'


class Rejected(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


class EgressBroker:
    """Policy gate, not a socket firewall. Transport is injected by trusted code."""
    def __init__(self, domains, transport=None):
        self.domains = frozenset(domains)
        self.transport = transport

    def request(self, url):
        try:
            parsed = urlsplit(url)
            allowed = (parsed.scheme == 'https' and parsed.hostname in self.domains
                       and parsed.port in (None, 443) and parsed.username is None
                       and parsed.password is None and not parsed.fragment)
        except ValueError:
            allowed = False
        if not allowed:
            raise Rejected('egress_denied')
        if self.transport is None:
            return {'decision': 'allow', 'transport': 'not_configured'}
        # A production transport MUST disable redirects and validate resolved IPs.
        return self.transport(url)


class SandboxService:
    def __init__(self, *, rehearsal=False, policy_path=None):
        self.policy = yaml.safe_load(Path(policy_path or ROOT / 'config/sandbox.yml').read_text())
        self.rehearsal = rehearsal
        self.lock = threading.Lock()
        self.active = {}
        self.runs = {}
        self.cost = {}
        self.broker = EgressBroker(self.policy['broker']['allowed_domains'])

    def readiness(self):
        if not self.rehearsal:
            raise Rejected('isolation_unavailable')
        for name in ('RLIMIT_CPU', 'RLIMIT_AS', 'RLIMIT_FSIZE', 'RLIMIT_NOFILE'):
            if not hasattr(resource, name):
                raise Rejected('isolation_unavailable')
        if not FIXTURE.is_file():
            raise Rejected('isolation_unavailable')
        return {'backend': 'rehearsal', 'resource_api': 'available',
                'container_isolation_verified': False}

    def browser_readiness(self):
        result = self.readiness()
        for path in ('anchor_ai/browser_tools.py', 'config/browser_policy.yml',
                     'labs/browser-agents/support_portal.html'):
            if not (ROOT / path).is_file():
                raise Rejected('browser_fixture_unavailable')
        return {**result, 'browser': 'offline_fake', 'real_browser_verified': False,
                'downloads_required': False}

    @contextlib.contextmanager
    def admit(self, tenant_id):
        if type(tenant_id) is not int or tenant_id <= 0:
            raise Rejected('invalid_tenant')
        caps = self.policy['admission']
        with self.lock:
            if (sum(self.active.values()) >= caps['global_active']
                    or self.active.get(tenant_id, 0) >= caps['tenant_active']):
                raise Rejected('capacity_unavailable')
            if (self.runs.get(tenant_id, 0) >= caps['tenant_runs']
                    or self.cost.get(tenant_id, 0) + 1 > caps['tenant_cost_units']):
                raise Rejected('tenant_quota')
            self.active[tenant_id] = self.active.get(tenant_id, 0) + 1
            self.runs[tenant_id] = self.runs.get(tenant_id, 0) + 1
            self.cost[tenant_id] = self.cost.get(tenant_id, 0) + 1
        try:
            yield
        finally:
            with self.lock:
                self.active[tenant_id] -= 1

    def execute_fixture(self, tenant_id, mode, *, wall_seconds=None):
        """Internal test/rehearsal API. Never expose fixture modes to the model."""
        self.readiness()
        browser_modes = {'browser_clean', 'browser_injected', 'browser_account'}
        if mode not in {'diagnostic', 'cpu', 'memory', 'file', 'fds', 'sleep', 'output', 'env', 'group'} | browser_modes:
            raise Rejected('unapproved_script')
        fixture = FIXTURE
        if mode in browser_modes:
            self.browser_readiness()
            fixture = ROOT / 'anchor_ai/browser_tools.py'
        with self.admit(tenant_id):
            limits = dict(self.policy['limits'])
            deadline = min(wall_seconds, limits['wall_seconds']) if wall_seconds is not None else limits['wall_seconds']
            if deadline <= 0:
                raise Rejected('invalid_deadline')
            with tempfile.TemporaryDirectory(prefix='anchor-sandbox-') as scratch:
                with tempfile.TemporaryFile() as output:
                    process = subprocess.Popen(
                        [sys.executable, '-I', str(Path(__file__).resolve()),
                         '--child', str(fixture), mode, json.dumps(limits)],
                        cwd=scratch, env={}, stdin=subprocess.DEVNULL,
                        stdout=output, stderr=output, start_new_session=True,
                        close_fds=True)
                    reason = 'exited'
                    try:
                        process.wait(timeout=deadline)
                    except subprocess.TimeoutExpired:
                        reason = 'wall_timeout'
                    finally:
                        # Also remove descendants left behind after an ordinary exit.
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait()
                    output.seek(0)
                    data = output.read(limits['output_bytes'] + 1)
                    if len(data) > limits['output_bytes']:
                        reason = 'output_limit'
                    return SandboxResult(exit_reason=reason,
                                         output=data[:limits['output_bytes']].decode(errors='replace'),
                                         returncode=process.returncode).model_dump()

    async def run_browser(self, *, scope, order_id='A100'):
        from anchor_ai.browser_tools import BrowserArguments, BrowserResult
        BrowserArguments(order_id=order_id)
        if getattr(scope, 'role', None) != 'support_agent':
            return BrowserResult(outcome='browser_unauthorized').model_dump()
        # Only the approved A100 rehearsal ticket is exposed by this backend.
        # Production must bind portal credentials and order ownership to scope.
        if order_id != 'A100':
            return BrowserResult(outcome='browser_order_unavailable').model_dump()
        try:
            result = await asyncio.to_thread(self.execute_fixture, scope.tenant_id, 'browser_clean')
            if result['exit_reason'] != 'exited' or result['returncode'] != 0:
                return BrowserResult(outcome='sandbox_failure').model_dump()
            data = json.loads(result['output'])
            return BrowserResult(outcome=data['outcome'], status=data['status']).model_dump()
        except Rejected as error:
            return BrowserResult(outcome=error.code).model_dump()

    async def run(self, *, scope, script_id='ticket_diagnostic', confirmed=False, dry_run=True):
        from anchor_ai.observability import span
        with span('python.sandbox', stage='tool', tenant_id=scope.tenant_id,
                  conversation_id=scope.conversation_id, sandbox_backend='rehearsal' if self.rehearsal else 'production') as audit:
            try:
                if script_id != 'ticket_diagnostic':
                    raise Rejected('unapproved_script')
                if getattr(scope, 'role', None) != 'support_agent':
                    raise Rejected('sandbox_unauthorized')
                self.readiness()
                if dry_run:
                    result = SandboxResult(exit_reason='dry_run').model_dump()
                elif not confirmed:
                    raise Rejected('confirmation_required')
                else:
                    result = await asyncio.to_thread(self.execute_fixture, scope.tenant_id, 'diagnostic')
                audit.set(sandbox_exit_reason=result['exit_reason'],
                          sandbox_output_bytes=len(result['output'].encode()))
                return result
            except Rejected as error:
                audit.set(sandbox_exit_reason=error.code, sandbox_output_bytes=0)
                audit.fail(error.code)
                return SandboxResult(exit_reason=error.code,
                                     backend='rehearsal' if self.rehearsal else 'production').model_dump()


def child():
    import runpy
    fixture, mode, raw = sys.argv[2:]
    limits = json.loads(raw)
    for constant, value in ((resource.RLIMIT_CPU, limits['cpu_seconds']),
                            (resource.RLIMIT_AS, limits['address_space_bytes']),
                            (resource.RLIMIT_FSIZE, limits['file_bytes']),
                            (resource.RLIMIT_NOFILE, limits['open_files'])):
        resource.setrlimit(constant, (value, value))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    sys.argv = [fixture, mode]
    runpy.run_path(fixture, run_name='__main__')


if __name__ == '__main__':
    if '--child' in sys.argv:
        child()
    else:
        service = SandboxService(rehearsal=True)
        print(json.dumps(service.readiness()))
        print(json.dumps(service.execute_fixture(1, 'diagnostic')))
