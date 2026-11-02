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
  echo 'PASS: offline endpoint parity, golden evaluation and tests; production deployment NOT approved'
  exit 0
fi
: "${ANCHOR_EVAL_EVIDENCE:?Set the trusted production evaluation bundle path}"
python3 -c 'from eval.release_gate import manifest, require_evidence; require_evidence(manifest())'
: "${FLY_APP:?Set FLY_APP to the reviewed Fly application}"
fly deploy --app "$FLY_APP" --config ops/deploy/fly.rag.toml
