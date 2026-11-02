#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
exec python3 ops/backups/verify_snapshot.py drill "$@"
