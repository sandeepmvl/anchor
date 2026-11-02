#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
mode="${1:---verify-only}"
case "$mode" in
  --verify-only|--deploy) ;;
  *) echo 'usage: rag_release.sh [--verify-only|--deploy]' >&2; exit 2 ;;
esac
export ANCHOR_LLM_PROVIDER=offline
export ANCHOR_EVAL_PROVIDER=offline
export ANCHOR_QUERY_PROVIDER=offline
export ANCHOR_EMBEDDING_PROVIDER=offline
python3 -m eval.eval_endpoint --testclient
python3 -m eval.rag_gate
python3 -m pytest -q anchor_ai/tests
if [ "$mode" = '--verify-only' ]; then
  echo 'PASS: endpoint parity, golden evaluation and existing tests; deployment stage permitted'
  exit 0
fi
: "${FLY_APP:?Set FLY_APP to the reviewed Fly application}"
fly deploy --app "$FLY_APP" --config ops/deploy/fly.rag.toml
