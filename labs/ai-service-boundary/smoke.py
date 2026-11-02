import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[2]
with socket.socket() as probe:
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]

env = dict(os.environ)
env.update(
    ANCHOR_AI_SECRET=secrets.token_hex(24),
    ANCHOR_AI_PROVIDER="offline",
    ANCHOR_LLM_PROVIDER="http",
    ANCHOR_AI_URL=f"http://127.0.0.1:{port}/v1/infer",
    ANCHOR_AI_OPEN_TIMEOUT="1",
    ANCHOR_AI_READ_TIMEOUT="2",
    RAILS_ENV="test",
)
runtime = None
rails = None


def stop(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


try:
    runtime = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1",
         "--port", str(port), "--log-level", "error"],
        cwd=root / "anchor_ai", env=env,
    )
    deadline = time.monotonic() + 3
    while True:
        if runtime.poll() is not None:
            raise RuntimeError("Uvicorn exited before becoming ready")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                break
        except OSError:
            if time.monotonic() >= deadline:
                raise RuntimeError("Uvicorn did not become ready")
            time.sleep(0.05)
    rails = subprocess.Popen(
        ["ruby", "bin/rails", "runner", "labs/ai-service-boundary/smoke.rb"],
        cwd=root, env=env, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, text=True,
    )
    for line in rails.stdout:
        if line.strip() == "STOP_RUNTIME":
            stop(runtime)
            rails.stdin.write("STOPPED\n")
            rails.stdin.flush()
        else:
            print(line, end="", flush=True)
    if rails.wait(timeout=2) != 0:
        raise RuntimeError("Rails smoke check failed")
finally:
    stop(rails)
    stop(runtime)
