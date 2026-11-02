import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from types import SimpleNamespace

import pytest
from anchor_ai import observability
from anchor_ai.runtime import BoundedRuntime, ContextError, run_diagnostic_ticket
from anchor_ai.sandbox import ROOT, Rejected, SandboxService


@pytest.fixture
def service():
    return SandboxService(rehearsal=True)


@pytest.mark.parametrize('mode,marker', [('memory', 'MEMORY_LIMIT'),
                                         ('file', 'FILE_LIMIT'), ('fds', 'FD_LIMIT')])
def test_resource_failures(service, mode, marker):
    result = service.execute_fixture(1, mode)
    assert result['exit_reason'] == 'exited'
    assert result['returncode'] == 0
    assert marker in result['output']


def test_resource_cpu(service):
    result = service.execute_fixture(1, 'cpu')
    assert result['exit_reason'] == 'exited'
    assert result['returncode'] in (-signal.SIGKILL, -signal.SIGXCPU)


def test_environment_scrubbed(service, monkeypatch):
    monkeypatch.setenv('ANCHOR_TEST_SECRET', 'must-not-reach-child')
    result = service.execute_fixture(1, 'env')
    assert json.loads(result['output']) == {'secret_present': False}


def test_output_bounded(service):
    result = service.execute_fixture(1, 'output')
    assert len(result['output'].encode()) <= service.policy['limits']['output_bytes']
    assert len(result['output']) == 4096


def test_deadline_process_group(service, monkeypatch, tmp_path):
    # Preserve scratch until after the descendant would have written its marker.
    import tempfile
    import contextlib
    @contextlib.contextmanager
    def scratch(**kwargs):
        yield str(tmp_path)
    monkeypatch.setattr(tempfile, 'TemporaryDirectory', scratch)
    start = time.monotonic()
    result = service.execute_fixture(1, 'group', wall_seconds=0.2)
    assert result['exit_reason'] == 'wall_timeout'
    assert 'CHILD_STARTED' in result['output']
    assert time.monotonic() - start < 1.5
    time.sleep(0.55)
    assert not (tmp_path / 'escaped-child').exists()
    assert service.active[1] == 0


def test_egress_policy(service):
    calls = []
    service.broker.transport = lambda url: calls.append(url) or {'status': 200}
    for url in ('https://attacker.example/x', 'https://kb.anchor.example.attacker.example',
                'http://kb.anchor.example', 'https://kb.anchor.example:444',
                'https://user@kb.anchor.example', 'https://127.0.0.1',
                'https://kb.anchor.example/#fragment'):
        with pytest.raises(Rejected, match='egress_denied'):
            service.broker.request(url)
    assert calls == []
    assert service.broker.request('https://kb.anchor.example/read') == {'status': 200}
    assert calls == ['https://kb.anchor.example/read']


def test_production_fails_closed(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('A production request must not spawn a process')
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    with pytest.raises(Rejected, match='isolation_unavailable'):
        SandboxService().execute_fixture(1, 'diagnostic')


def test_capacity_and_quota(service, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Admission rejection must precede process creation')
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    with service.admit(1):
        with pytest.raises(Rejected, match='capacity_unavailable'):
            service.execute_fixture(1, 'diagnostic')
        with service.admit(2), service.admit(3), service.admit(4):
            with pytest.raises(Rejected, match='capacity_unavailable'):
                service.execute_fixture(5, 'diagnostic')
    assert sum(service.active.values()) == 0
    service.policy['admission']['tenant_runs'] = 1
    with pytest.raises(Rejected, match='tenant_quota'):
        service.execute_fixture(1, 'diagnostic')
    service.policy['admission']['tenant_cost_units'] = 0
    with pytest.raises(Rejected, match='tenant_quota'):
        service.execute_fixture(9, 'diagnostic')


def test_configuration():
    service = SandboxService()
    production = service.policy['production']
    assert production['network'] == 'none'
    assert production['read_only'] is True
    assert production['host_mounts'] == []
    assert production['cap_drop'] == ['ALL']
    assert production['no_new_privileges'] is True
    assert production['pids_limit'] == 50
    assert production['memory'] == '256m'
    assert production['cpus'] == 0.5
    assert production['user'] == '65532:65532'
    dockerfile = (ROOT / 'ops/sandbox/Dockerfile').read_text()
    assert 'USER 65532:65532' in dockerfile
    assert 'RUN ' not in dockerfile
    profile = json.loads((ROOT / production['seccomp']).read_text())
    assert profile['defaultAction'] == 'SCMP_ACT_ERRNO'
    allowed = {name for rule in profile['syscalls']
               if rule['action'] == 'SCMP_ACT_ALLOW' for name in rule['names']}
    assert not allowed.intersection({'mount', 'umount2', 'ptrace', 'unshare', 'setns',
                                     'clone', 'clone3', 'bpf', 'keyctl', 'init_module',
                                     'finit_module', 'delete_module', 'reboot'})
    runbook = (ROOT / 'ops/runbooks/sandbox_capacity.md').read_text()
    assert '50 concurrent' in runbook
    assert 'shared admission' in runbook


class LocalTicketRuntime:
    """Injected offline identity fixture, not a production token verifier."""
    def identity(self, token):
        if token != 'local-ticket':
            raise ContextError('unverified_execution_context')
        return SimpleNamespace(tenant_id=1, conversation_id=7, role='support_agent')


def test_diagnostic_replay(service, monkeypatch):
    records = []
    monkeypatch.setattr(observability, 'SINK', records.append)
    runtime = LocalTicketRuntime()
    async def exercise():
        preview = await run_diagnostic_ticket(runtime, 'local-ticket', service)
        assert preview['exit_reason'] == 'dry_run'
        assert service.runs == {}
        refused = await run_diagnostic_ticket(runtime, 'local-ticket', service, dry_run=False)
        assert refused['exit_reason'] == 'confirmation_required'
        result = await run_diagnostic_ticket(runtime, 'local-ticket', service,
                                             dry_run=False, confirmed=True)
        assert result['exit_reason'] == 'exited'
        assert json.loads(result['output'])['status'] == 'input_valid'
        with pytest.raises(ContextError):
            await run_diagnostic_ticket(runtime, 'forged', service, confirmed=True, dry_run=False)
        return result
    result = asyncio.run(exercise())
    sandbox = [r for r in records if r['name'] == 'python.sandbox']
    agents = [r for r in records if r['name'] == 'python.diagnostic_agent']
    assert sandbox[-1]['trace_id'] == agents[-1]['trace_id']
    assert sandbox[-1]['parent_span_id'] == agents[-1]['span_id']
    assert sandbox[-1]['sandbox_exit_reason'] == 'exited'
    assert sandbox[-1]['sandbox_output_bytes'] <= 4096
    assert all('output' not in r and 'script' not in r for r in records)
    print('Anchor diagnostic ticket: approved script executed; bounded output; audited exit_reason=exited; rehearsal only')
    print(json.dumps(result))


def test_identity_checked_before_diagnostic(service):
    # Exercise the existing runtime identity implementation without booting a DB.
    runtime = object.__new__(BoundedRuntime)
    runtime.verify = lambda token: SimpleNamespace(tenant_id=1)
    with pytest.raises(ContextError, match='unverified_execution_context'):
        asyncio.run(run_diagnostic_ticket(runtime, 'forged', service))
    assert service.runs == {}
