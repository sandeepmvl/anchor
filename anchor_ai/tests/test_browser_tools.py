import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from anchor_ai.browser_tools import (
    BrowserArguments, Executor, FakeDriver, POLICY, ROOT, URL,
    check_action, run_agent,
)
from anchor_ai.sandbox import Rejected, SandboxService
from anchor_ai.tools import MemoryOrderBackend, register_browser, registry_for


def test_clean_status_and_redacted_observations():
    driver = FakeDriver()
    result = run_agent(driver)
    assert result['outcome'] == 'result'
    assert result['status'] == 'Delivered'
    assert [a['kind'] for a in driver.calls] == ['navigate', 'type', 'click', 'read']
    assert 'alex@example.test' in driver.observe()['dom_text']
    trace = json.dumps(result['trace'])
    assert 'alex@example.test' not in trace
    assert '202 555' not in trace
    assert 'dom_text' not in trace
    screenshots = [e for e in result['trace'] if e['event'] == 'SCREENSHOT']
    assert screenshots and all(e['redacted'] for e in screenshots)
    assert all(e['representation'] == 'state_record' for e in screenshots)


def test_injected_navigation_never_reaches_driver():
    driver = FakeDriver('injected')
    result = run_agent(driver)
    assert result['outcome'] == 'blocked'
    denial = result['trace'][-1]
    assert denial['step'] == 2
    assert denial['layer'] == 'ui_allowlist'
    assert denial['rule'] == 'domain_not_allowed'
    assert denial['domain'] == 'steal.example'
    assert denial['action']['kind'] == 'navigate'
    assert driver.calls == [{'kind': 'navigate', 'url': URL}]
    assert 'alex@example.test' not in json.dumps(result['trace'])


@pytest.mark.parametrize('url', [
    'http://127.0.0.1:8766/support_portal.html',
    'http://127.0.0.1.evil:8765/support_portal.html',
    'http://user@127.0.0.1:8765/support_portal.html',
    'http://127.0.0.1:8765/admin',
    'file:///etc/passwd', 'https://steal.example/',
])
def test_exact_origin_and_path(url):
    driver = FakeDriver()
    executor = Executor(driver)
    assert executor.execute({'kind': 'navigate', 'url': url}, 1)[0] == 'blocked'
    assert driver.calls == []


def test_dataset_cannot_grant_permissions():
    cases = ROOT / 'eval/datasets/browser_injection_cases.jsonl'
    for line in cases.read_text().splitlines():
        case = json.loads(line)
        driver = FakeDriver()
        driver.url = URL
        executor = Executor(driver)
        outcome, _ = executor.execute(case['action'], 1)
        if case['expected'] == 'approval_required':
            assert outcome == 'approval_required'
            assert executor.trace[-1]['event'] == 'APPROVAL_REQUIRED'
        else:
            assert outcome == 'blocked'
            assert executor.trace[-1]['rule'] == case['expected']
        assert not driver.calls


def test_every_operation_has_a_pre_dispatch_check():
    for kind in ('type', 'click', 'submit', 'read'):
        driver = FakeDriver()
        executor = Executor(driver)
        assert executor.execute({'kind': kind, 'target': 'status'}, 1)[0] == 'blocked'
        assert not driver.calls
    for action in ({'kind': 'submit', 'target': 'lookup'},
                   {'kind': 'click', 'target': 'download'},
                   {'kind': 'type', 'target': 'order', 'value': 'private@example.test'}):
        driver = FakeDriver()
        driver.url = URL
        assert Executor(driver).execute(action, 1)[0] == 'blocked'
        assert not driver.calls
    assert FakeDriver().attempt_download() == 'download_blocked_simulation'


def test_consequential_click_requires_trusted_bound_one_use_approval():
    driver = FakeDriver()
    driver.url = URL
    action = {'kind': 'click', 'target': 'account_change', 'approved': True}
    executor = Executor(driver)
    assert check_action(action, URL) is None
    assert executor.execute(action, 1)[0] == 'approval_required'
    assert not driver.calls
    grants = {(1, 7, URL, 'click', 'account_change')}
    def consume(binding):
        if binding not in grants:
            return False
        grants.remove(binding)
        return True
    wrong_scope = Executor(driver, tenant_id=2, approval=consume)
    assert wrong_scope.execute(action, 2)[0] == 'approval_required'
    executor.approval = consume
    assert executor.execute(action, 3)[0] == 'continue'
    assert executor.execute(action, 4)[0] == 'approval_required'
    assert len(driver.calls) == 1


def test_network_simulation_is_not_allowlist_denial():
    result = run_agent(FakeDriver(network_denied=True))
    assert result['outcome'] == 'network_error'
    assert result['trace'][-2]['event'] == 'ALLOWED'
    error = result['trace'][-1]
    assert error['event'] == 'NETWORK_ERROR'
    assert error['layer'] == 'sandbox_egress_simulation'
    assert 'rule' not in error


def test_sandbox_worker_and_real_deadline():
    service = SandboxService(rehearsal=True)
    readiness = service.browser_readiness()
    assert readiness['real_browser_verified'] is False
    assert readiness['container_isolation_verified'] is False
    wrapped = service.execute_fixture(1, 'browser_clean')
    assert wrapped['returncode'] == 0, wrapped['output']
    assert json.loads(wrapped['output'])['status'] == 'Delivered'
    assert service.active[1] == 0
    timed = SandboxService(rehearsal=True).execute_fixture(1, 'sleep', wall_seconds=0.01)
    assert timed['exit_reason'] == 'wall_timeout'
    assert timed['returncode'] != 0
    # The existing fixture actually attempts excessive file descriptors.
    limited = SandboxService(rehearsal=True).execute_fixture(1, 'fds')
    assert limited['exit_reason'] in {'exited', 'output_limit'}
    assert len(limited['output'].encode()) <= service.policy['limits']['output_bytes']
    with pytest.raises(Rejected):
        SandboxService().browser_readiness()


def test_registry_keeps_prior_tools_and_constrains_arguments():
    old = registry_for(MemoryOrderBackend())
    service = SandboxService(rehearsal=True)
    new = register_browser(old, service)
    assert set(new) == set(old) | {'browser_order_status'}
    assert all(new[name] is old[name] for name in old)
    with pytest.raises(ValidationError):
        BrowserArguments(order_id='A100', approved=True)
    scope = SimpleNamespace(tenant_id=1, conversation_id=7, role='support_agent')
    result = asyncio.run(new['browser_order_status'].run(scope=scope, order_id='A100'))
    assert result == {'outcome': 'result', 'status': 'Delivered', 'backend': 'rehearsal'}
    bad = SimpleNamespace(tenant_id=1, conversation_id=7, role='customer')
    assert asyncio.run(service.run_browser(scope=bad))['outcome'] == 'browser_unauthorized'


def test_production_configuration_without_claiming_container_execution():
    dockerfile = (ROOT / 'ops/sandbox/Dockerfile').read_text()
    assert 'playwright/python:v1.58.0-noble' in dockerfile
    assert 'playwright==1.58.0' in dockerfile
    assert 'USER 65532:65532' in dockerfile
    assert '/opt/diagnostic.py' in dockerfile
    assert 'support_portal' not in dockerfile or 'labs/browser-agents/' in dockerfile
    assert 'labs/browser-agents/' in dockerfile
    assert 'playwright install' not in dockerfile
    assert '--no-sandbox' not in dockerfile
    profile = json.loads((ROOT / 'ops/sandbox/sandbox-seccomp.json').read_text())
    assert profile['defaultAction'] == 'SCMP_ACT_ERRNO'
    allowed = {name for entry in profile['syscalls']
               if entry['action'] == 'SCMP_ACT_ALLOW' for name in entry['names']}
    assert {'read', 'write', 'execve', 'clone', 'bind', 'listen', 'seccomp'} <= allowed
    assert not {'mount', 'ptrace', 'bpf', 'reboot', 'kexec_load'} & allowed
    source = (ROOT / 'anchor_ai/browser_tools.py').read_text()
    assert "accept_downloads=False" in source
    assert "service_workers='block'" in source
    assert "screenshot(mask=" in source
    assert 'download.cancel()' in source
    assert POLICY['writes']['submit'] == {}
