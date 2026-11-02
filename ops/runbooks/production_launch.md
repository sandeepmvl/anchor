# Anchor production launch

## Required local rehearsal

Run python3 bin/deploy production --offline from the repository root.
This validates the composed production topology and runs local launch gates.
It does NOT start the isolated production-topology containers.
The receipt is .git/anchor-launch/rehearsal.json, not production authorization.
Run bash ops/load/load_test.sh --target production --offline separately.
The target alias resolves only to the existing in-process benchmark.
Run python3 bin/deploy production --offline --reject-candidate after a clean run.
A rejected candidate must leave the previous receipt byte-for-byte unchanged.
This is pointer-retention evidence, not a live rollback drill.

## Existing product evidence

Run ruby test/integration/production_launch_test.rb --deployed on prepared infrastructure.
Despite its flag name, this runs application integration tests, not live HTTP probes.
Retain allowed and denied admin, agent and end-customer requests and tenant probes.
Retain quota denials, redacted logs, guarded tool results and full-context handoff traces.
Run the complete bin/release_gate with ANCHOR_PRODUCTION_LAUNCH=1.
It includes Rails, Python, retrieval, RAG, agent, multi-agent, endpoint red-team,
security and consolidated golden/calibration/trajectory/regression gates.
Document and customer-message injection cases must both pass for this candidate.
Smoke and sample results never authorize launch.

## On your machine: real infrastructure only

The current validate_manifest.release_gate intentionally refuses production release.
Do not delete that refusal or substitute a hand-written success JSON file.
A trusted infrastructure adapter is still required before this workflow can ship.
Use compose.yml plus ops/deploy/anchor-production.yml for the isolated rehearsal stack.
Use authorized synthetic tenants and existing chat, email and voice fixture flows.
Measure sustained offered load, completions, rejections, errors, queue waits and p95/p99.
Inspect scoped authorization, quota decisions, release provenance and tool spans.
Execute ops/deploy/rollback_pipeline.sh for compatible prompt/model/index versions.
Execute ops/backups/drop_and_restore_drill.sh only against an isolated restore target.
Compare restored tenant records, conversation history, indexes and action audit records.
Disable autonomy after queue admission and verify the tool boundary performs no side effect.
Verify zero-downtime application rollout and previous-release availability independently.
ANCHOR_LLM_PROVIDER=openai selects the existing live provider only with approved credentials.
A provider flag does not supply deployment, judge, recovery or capacity evidence.

## Production readiness sign-off — complete before tagging

Record candidate commit and bundle, prompt/model/index versions, evidence hashes and URLs.
Record load objectives and results, all gate results, recovery results and on-call owner.
Record reviewer identity, decision and UTC timestamp through the organization's approval system.
Only after trusted deployment evidence and review pass, create a signed immutable release tag.
Never force-move a production tag. An annotated unsigned tag is not a cryptographic signature.
