#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ "$*" != "--target production --offline" || "${ANCHOR_LOCUST:-0}" == "1" ]]; then
  echo 'Load refused: use --target production --offline; live target approval adapter is not installed.' >&2
  exit 1
fi
exec "${ANCHOR_PYTHON:-python3}" -c 'import asyncio; from ops.load.ticket_spike import production_rehearsal; asyncio.run(production_rehearsal("production"))'
