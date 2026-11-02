import errno
import json
import os
from pathlib import Path
import socket
import sys
import time

mode = sys.argv[1]
if mode == 'diagnostic':
    print(json.dumps({'ticket': 'diagnostic', 'status': 'input_valid', 'records': 1}))
elif mode == 'cpu':
    while True:
        pass
elif mode == 'memory':
    try:
        bytearray(128 * 1024 * 1024)
    except MemoryError:
        print('MEMORY_LIMIT', flush=True)
    else:
        raise AssertionError('Memory allocation unexpectedly succeeded')
elif mode == 'file':
    try:
        with open('oversize', 'wb', buffering=0) as target:
            for _ in range(16):
                target.write(b'x' * 4096)
    except OSError as error:
        assert error.errno == errno.EFBIG
        print('FILE_LIMIT', flush=True)
    else:
        raise AssertionError('File limit did not trip')
elif mode == 'fds':
    handles = []
    try:
        for _ in range(100):
            handles.append(open('/dev/null'))
    except OSError as error:
        assert error.errno == errno.EMFILE
        print('FD_LIMIT', flush=True)
    else:
        raise AssertionError('Descriptor limit did not trip')
elif mode == 'sleep':
    time.sleep(30)
elif mode == 'group':
    child = os.fork()
    if child == 0:
        time.sleep(0.5)
        Path('escaped-child').write_text('bad')
        os._exit(0)
    print('CHILD_STARTED', flush=True)
    time.sleep(30)
elif mode == 'output':
    try:
        os.write(1, b'x' * 16384)
        os.write(1, b'y' * 16384)
    except OSError:
        pass
elif mode == 'env':
    print(json.dumps({'secret_present': 'ANCHOR_TEST_SECRET' in os.environ}))
elif mode == 'socket':
    # Container-only: no connection is made by the required offline tests.
    try:
        socket.create_connection(('192.0.2.1', 443), timeout=1).close()
    except OSError as error:
        print(json.dumps({'socket': 'failed', 'errno': error.errno}), flush=True)
    else:
        raise AssertionError('Unexpected outbound connection')
elif mode == 'fork':
    # Container-only. Refuse to fork without the expected cgroup ceiling.
    maximum = Path('/sys/fs/cgroup/pids.max').read_text().strip()
    assert maximum == '50', 'Run only inside the configured container'
    events = Path('/sys/fs/cgroup/pids.events')
    def count():
        return int(dict(line.split() for line in events.read_text().splitlines())['max'])
    before = count()
    children = []
    for _ in range(80):
        try:
            pid = os.fork()
        except OSError as error:
            assert error.errno == errno.EAGAIN
            assert count() > before
            print(json.dumps({'fork': 'denied', 'pids_max_delta': count() - before}), flush=True)
            break
        if pid == 0:
            time.sleep(60)
            os._exit(0)
        children.append(pid)
    else:
        raise AssertionError('PID ceiling did not trip')
    time.sleep(60)  # The external supervisor must stop the entire container.
else:
    raise ValueError('Unknown fixture mode')
