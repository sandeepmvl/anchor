#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
# This executable lab path uses the real registry with a local SQLite fixture DB.
# Production requires a separately reviewed readiness/evidence importer.
exec ruby labs/rollback-strategies/registry_demo.rb "$@"
