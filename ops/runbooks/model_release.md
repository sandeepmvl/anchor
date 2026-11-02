# Independent model releases

Course project: Anchor.

## Scope
The original model release controller selects constructed offline recordings only. It never enables a live model. Its independent selection API remains available for existing callers. Bundle-aware requests instead pin a complete prompt/model/index release through AiReleaseRegistry and must not reread independent pointers during that request.

## Atomic bundle recovery — lesson 3759
The shared ai_release_pointers row is keyed by tenant and answer channel. Admission reads the routing fields and immutable bundle map in one SELECT. Stable conversation hashing determines the candidate slice. A generation check rejects stale writers; a row lock serializes changes. Pointer, candidate invalidation, and audit history commit in one database transaction. SQLite serializes writes in the local exercise; production must use the application's shared durable database and retry transient database-lock/deadlock failures without bypassing generation checks.

Register immutable artifacts before staging. Apply the migration through normal Rails deployment. Publish prompt versions with PromptRegistry.publish; verify model availability and resident index readiness before admitting traffic. Wrap synchronous Rails admission with registry.with_request and pass its exported snapshot to the Python worker. Queued jobs must persist that snapshot, not a mutable alias. Python calls serving_artifacts.pin_release_bundle once, then uses the returned prompt/model/index and snapshot_for_release for the cache. Existing routes are not automatically rewired by this lesson; wire admission explicitly before enabling bundle routing. Never globally cache a request-pinned object.

The required offline exercise uses the actual model, migration and registry through labs/rollback-strategies/registry_demo.rb. Run the rollback_pipeline.sh stages shown in the lesson. The local .anchor-release-demo.sqlite3 file holds exercise state only. This runner uses a deterministic fixture adapter and no live-provider environment switch: ANCHOR_LLM_MODE=live still belongs to the existing provider adapter, not to this release simulator. A production adapter must consume the pinned immutable model identifier and preserve tenant authorization and safe-tool controls.

Promotion requires sufficient completed control/candidate scores, a follow-up-sized window, absolute quality/error/latency limits and a relative quality limit. The fixture golden and recorded-judge composition runs through eval.release_gate and is bound to the full bundle. These are constructed scores, not production evidence. This registry deliberately refuses all non-offline selection until a trusted readiness/evaluation importer is implemented; setting a JSON flag or provider variable cannot open that gate.

Revert clears candidate assignments for all new turns. Existing work retains its original bundle and cache namespace. For a promoted candidate, revert restores the prior compatible bundle. It does not cancel streaming work or reverse tool actions. Keep confirmation, authorization and idempotency controls in place.

For an index-only incident, register a new compatible bundle retaining prompt and model and selecting the already-built prior resident index. Promote or revert only its pointer; do not rebuild, copy or restore index contents in the incident path. Retain tenant filters. If a prior bundle requires a removed schema/tool, refuse revert, disable unsafe actions and roll forward or restore a compatible shim first.

Before production use, collect trusted golden and versioned-judge evidence, candidate/control health by actual served identifiers, live readiness for both targets, and security regressions. Measure fleet acknowledgment and propagation, not just database commit time. This lab verifies neither live inference nor authenticated HTTP integration nor propagation across serving hosts.

## Trust and identity
Review config/model_releases.yml in source control. Its expected JSON contents define canonical SHA-256 identities. Promotion compares candidate artifacts to those expectations and reruns the recorded gate. Do not edit the manifest to excuse an unexpected hash. Publish artifacts immutably; keep adapters and their original configs alongside any merged export.

## Selection
The default state path is .git/anchor-model-release.json for this local lab. The deploy command holds a file lock, writes and fsyncs a temporary file, atomically replaces state, and fsyncs its directory. A production coordinator needs a shared durable store and equivalent compare-and-swap semantics across hosts.

Set ANCHOR_MODEL_RELEASE_STATE to the absolute selection-file path to opt the existing offline configured_provider into release recordings. Leave it unset to retain its original behavior. Construct the provider once per request. Do not cache a pinned provider globally. ANCHOR_LLM_MODE=live uses the existing LiveProvider and its OPENAI_API_KEY; these commands never set live mode. The separate existing LlmClient uses ANCHOR_LLM_PROVIDER for its provider mode.

## Promotion and rollback
Run the adapter numeric walkthrough, the recorded model gate, and the accumulated tests before promotion. Promote with ops/deploy/model_release.py --offline --promote adapter-recorded. Roll back with the same command and --offline --rollback. Both operations reevaluate hash-bound evidence. A failed check leaves state unchanged. New requests observe the new selection; requests already pinned retain their reply and artifact identity.

## Real PEFT export
Reload the exact base revision in fp16/bf16. Load the original adapter config, matching tokenizer, vocabulary changes, and saved modules. Save the model returned by merge_and_unload. Retain the unmerged adapter for independent comparison and multi-adapter serving. Evaluate each quantized export separately; floating-point agreement is not quantized-artifact evidence.

## Production gate still required
Use independently held-out golden cases, the versioned LLM judge, safety checks, retrieval and agent regressions, and latency/cost measurements bound to the exact exported artifact. Compare against a separately and correctly loaded adapter. A paired 95% interval wholly below zero flags regression; an interval containing zero does not prove equivalence. Constructed recordings never satisfy a production deployment gate.

## Troubleshooting
A base-identity or hash error is a stop, not a warning. A wrong-alpha model can match its own merged output and still differ from the correct reference. A malformed state file fails closed. Do not delete state to conceal a failed promotion; investigate it and restore an approved selection through the release process.

# Preference alignment lesson 3688

## Preference-aligned candidates
The DPO manifest distinguishes a trainable policy from its frozen reference.
The proposed artifacts/merged-sft path is a configuration placeholder, not an
artifact created by this lab. Before real training, bind both initial copies
and the tokenizer to the reviewed merged export and immutable artifact hashes.
Never update the reference during DPO. Inspect completion masks and truncation.

Export only permitted, reviewed, tenant-scoped, redacted transcript comparisons.
Group related conversations before splitting and review near duplicates.
Email/phone pattern checks are a teaching minimum, not complete PII detection.
Use the existing PII review and defer controls before storage, logs, or export.

Run the recorded preference analyzer, independent resolved-ticket evaluation,
and customization safety gate. A justified handoff is appropriate even when it
is not a resolution. Never improve resolution numbers by inventing refunds.
Block privacy leakage, injection compliance, unsupported claims, unsafe tool
actions, and escalation regressions regardless of preference improvements.

Rising margins with worse independent outcomes flag proxy over-optimization.
Inspect generated replies, rejected-set coverage, and length-stratified scores.
Compare earlier checkpoints, higher beta, and reviewed generic negatives.
Changing beta changes raw margin scale; rerun independent evaluations.

These constructed recordings cannot authorize deployment. Retain the golden
and versioned LLM-judge gate, tenant-document and customer-message red-team
cases, retrieval checks, trajectory tests, Rails CI, and production artifact,
latency, and cost evidence. Do not promote dpo-recorded through the adapter
release controller or set a live model override from this report.

## Consolidated evidence milestone
Promotion and rollback now additionally require --evidence PATH or ANCHOR_EVAL_EVIDENCE.
The expected evaluation manifest includes the complete approved_entry target, binding the candidate,
model manifest hash and artifact hashes as well as commit, dataset, rubric, prompt, model and index.
All four reports must be complete, fresh and matching. Offline evidence permits only this controller's
offline selection exercise, never deployment. Missing or stale evidence leaves selection unchanged.
See evaluation_gate.md for trusted production provenance, baseline, ruleset and exception policy.
