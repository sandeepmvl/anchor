#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
exec python3 ops/drills/stale_release_pointer.py eval "$@"
