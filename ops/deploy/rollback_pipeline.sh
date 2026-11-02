#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
# Resync the CURRENT intended bundle; do not toggle previous/stable again.
if [ "${1:-}" = '--revert' ] && [ "${2:-}" = '--resync' ]; then
  shift 2
  exec python3 ops/drills/stale_release_pointer.py resync "$@"
fi
# This executable lab path uses the real registry with a local SQLite fixture DB.
# Production requires a separately reviewed readiness/evidence importer.
exec ruby labs/rollback-strategies/registry_demo.rb "$@"
