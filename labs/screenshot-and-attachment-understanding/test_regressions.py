"""Run all accumulated tests in bounded, disjoint groups; edits no files."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('suite', choices=['rails', 'python'])
parser.add_argument('shard', type=int)
parser.add_argument('--shards', type=int, default=4)
args = parser.parse_args()
if not 0 <= args.shard < args.shards:
    parser.error('shard must be inside the shard count')
root = Path(__file__).resolve().parents[2]
pattern = 'test/**/*_test.rb' if args.suite == 'rails' else 'anchor_ai/tests/**/test_*.py'
all_files = sorted(str(p.relative_to(root)) for p in root.glob(pattern))
files = all_files[args.shard::args.shards]
if not files:
    parser.error('empty shard')
env = os.environ.copy()
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    env[key] = '1'
env['PARALLEL_WORKERS'] = '1'
command = (['ruby', 'bin/rails', 'test'] if args.suite == 'rails'
           else [sys.executable, '-m', 'pytest', '-q'])
print(f'{args.suite} shard {args.shard + 1}/{args.shards}: '
      f'{len(files)}/{len(all_files)} files; no exclusions', flush=True)
result = subprocess.run(command + files, cwd=root, env=env)
raise SystemExit(result.returncode if result.returncode >= 0 else 1)
