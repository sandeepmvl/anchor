import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


def git(directory, *args):
    return subprocess.run(['git', '-C', str(directory), *args],
                          check=True, capture_output=True, text=True)


def main():
    root = Path.cwd()
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='anchor-interview-') as temporary:
        snapshot = Path(temporary) / 'snapshot'
        snapshot.mkdir()
        files = git(root, 'ls-files', '--cached', '--others', '--exclude-standard').stdout.splitlines()
        for name in files:
            source = root / name
            if source.is_file() and not source.is_symlink():
                target = snapshot / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        git(snapshot, 'init', '-q')
        git(snapshot, 'add', '.')
        git(snapshot, '-c', 'user.name=Interview', '-c', 'user.email=offline@example.invalid',
            '-c', 'commit.gpgsign=false', 'commit', '-qm', 'disposable interview snapshot')
        worktree = Path(temporary) / 'broken'
        git(snapshot, 'worktree', 'add', '--detach', str(worktree), 'HEAD')
        git(worktree, 'apply', '--unidiff-zero', 'labs/staff-interview/broken_escalation.patch')
        result = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-s',
            'anchor_ai/tests/test_bounded_runtime.py', '-k', 'staff_terminal_escalation_contract'],
            cwd=worktree, capture_output=True, text=True, timeout=7)
        print(result.stdout + result.stderr)
        expected_failure = ('FAILED anchor_ai/tests/test_bounded_runtime.py::'
                            'test_staff_terminal_escalation_contract[evidence]')
        if (result.returncode != 1
                or expected_failure not in result.stdout
                or '1 failed, 1 passed' not in result.stdout):
            raise SystemExit('Expected only the planted evidence-escalation case to fail; '
                             'the budget-escalation case must pass')
        print('Planted regression detected: missing terminal transition after empty evidence.')
        print(f'Diagnostic setup elapsed: {time.monotonic() - started:.2f}s; rehearsal target: 20 minutes.')
        print('Offline only: deployed authentication, live providers, container isolation, and distributed recovery not verified.')


if __name__ == '__main__':
    main()
