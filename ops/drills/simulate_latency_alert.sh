#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
# Preserve previous evidence instead of silently overwriting a failed attempt.
if [ -d tmp/anchor-operations ]; then
  archive="tmp/anchor-operations-$(date +%s)-$$"
  mv tmp/anchor-operations "$archive"
  printf 'Previous fixture evidence archived: %s\n' "$archive"
fi
exec ruby labs/production-operations/worker_recovery.rb start
