# Anchor RAG endpoint release

## Required offline evidence

Run `python3 -m eval.eval_endpoint --testclient` from the repository root.
Inspect `.git/anchor-endpoint-eval/{local,endpoint}-capture.json` and both score files.
Run `sh ops/deploy/rag_release.sh --verify-only` before publication.
The wrapper requires fresh endpoint parity, the existing golden gate and accumulated AI tests.
A nonzero status blocks publication. Never add `|| true` to these checks.

## Boundary and scope

POST /rag uses X-Anchor-Secret, configured by ANCHOR_AI_SECRET.
Rails authenticates the end user and supplies trusted tenant/conversation scope.
The release runtime accepts only the loaded ticket's tenant, generation and question.
It returns the full answer capture and an explicit scope envelope.
The pre-existing /v1/rag contract is unchanged.

## Recorded versus live search

The default release runtime reuses the existing offline gate's recorded search I/O.
It does not rebuild an index and does not establish live hybrid-index quality.
ANCHOR_LLM_PROVIDER is the existing provider switch; offline is the default.
Setting it to a live provider alone is intentionally insufficient for this endpoint.
The application composition must inject rag_release_runtime_factory with a ReleaseRuntime
whose search_factory supplies the existing tenant-scoped hybrid retriever and reranker.
Pin the index generation, embedding revision, reranker revision and library versions.
Run the full live-index retrieval evaluation before publishing that composition.
Never silently fall back from a configured live search to recorded evidence.

## Container check on a Docker-equipped workstation

Keep the Rails Dockerfile's default target unchanged.
Build the Python target with `docker build --target rag_service -t anchor-rag .`.
Run it with `docker run --rm -p 8080:8080 -e ANCHOR_AI_SECRET anchor-rag`.
Set ANCHOR_AI_SECRET in the calling shell first; do not commit it.
In another terminal run the evaluator with `--base-url http://127.0.0.1:8080`.
Stop the container after the check. This proves image contents and an actual TCP listener.
The required offline lab does not require Docker or leave a server running.

## Fly publication

Create the Fly app without deploying; choose its name explicitly.
Use ops/deploy/fly.rag.toml and set FLY_APP to that name.
Store ANCHOR_AI_SECRET with Fly secrets, not image ENV or source control.
Run the release wrapper with --deploy only after reviewing the runtime composition.
Replay endpoint evaluation with --base-url https://YOUR-APP.fly.dev.
Inspect build output for installation failures and platform logs for runtime failures.
Only hosted checks prove public reachability, VM limits, environment and wake-up behavior.

## Diagnose before changing retrieval

Diff per-ticket ordered evidence IDs and unrounded scores, not just aggregates.
Compare the artifact checksum, pinned packages and exact model revisions.
An auto-start timeout followed by successful startup differs from repeated failed readiness.
Check lifespan completion, health grace, memory exits and 0.0.0.0:8080 binding.
Missing artifacts, secrets and unreachable index hosts are deployment failures.
Use bounded caller timeouts; do not retry a failing service indefinitely.
