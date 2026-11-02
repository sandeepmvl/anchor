# Independent model releases

Course project: Anchor.

## Scope
This release controller selects constructed offline recordings only. It never enables a live model. Model selection is independent of Rails deployment, prompt selection, and retrieval-generation selection.

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
