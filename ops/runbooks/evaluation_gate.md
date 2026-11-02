# Anchor evaluation release gate

Course project: Anchor.

## Local scope
`ruby bin/release_gate --smoke` runs only the new gate contract tests and explicit fake refund control.
It does not verify the production judge, service database, endpoint composition, container isolation,
complete accumulated suite, or GitHub merge protection. The default wrapper retains the accumulated
Rails/Python, retrieval, RAG, agent, and multi-agent commands and then requires production evidence.
Do not relabel smoke output as production evidence.

## Composition adapter
Provision ANCHOR_EVAL_ADAPTER as a reviewed executable on the integration runner. Production commands
require OPENAI_API_KEY and refuse a missing adapter. The adapter receives operation, manifest path,
output path, and tier. Operations are candidate, golden, calibration, trajectory, and regression.
Candidate must stage the authorized tagged golden export in ANCHOR_EVAL_DATABASE_URL, validate its
identity, call the existing orchestrator with the checked-out prompt, and write redacted candidate.json.
Other operations read that candidate from .git/anchor-eval/candidate.json. Judge and trajectory inspect
the same recording. Regression independently replays the full existing mandatory safety set.
Reports contain manifest, created_at, status, finite score, synthetic example_id, and integer step.
Missing infrastructure and evaluator exceptions are failures, never synthetic successful reports.
The production adapter and its service DB client must be provisioned and tested on the real runner;
this lesson does not download or invent them in the offline session.

## Baseline and promotion trust
The lookup selects the newest non-expired anchor-baseline artifact from a successful eval-gate.yml
main run and requires full tier. Transport errors are fatal. Missing baseline blocks production.
Bootstrap the first full baseline in a separately reviewed administrator-run evaluation using the same
acceptance criteria; retain its run URL, manifest, reports, reviewer and artifact provenance. Do not
bootstrap from a PR or weaken the normal gate to permit absence. Schedule refreshes inside 14 days.
Release hosts fetch evidence only from trusted successful workflow runs, verify the run commit against
the reviewed release, and keep fetched files read-only to application users. A local JSON document is
not a signature or authority. Manifest hashes bind the bytes configured in config/eval_gate.yml;
production model/index configurations must pin actual artifact digests, not mutable aliases.
Independent prompt, model, and index releases need their own complete manifest and fresh evidence.
Promotion does not authorize a runtime composition that was not evaluated.

## Enforcement and exceptions
An administrator provisions an active main ruleset requiring exactly `Anchor release gate`.
Protect workflows, adapter provisioning, policy, manifest inputs and quarantine with code-owner review.
Verify blocking on a disposable deliberately failing PR and retain the ruleset export and run URL.
Quarantine only calibration/trajectory scoring faults after one diagnostic rerun on unchanged evidence.
Record distinct owner and reviewer, issue URL, and expires_at no more than seven days ahead in the
reviewed config list. Jobs still execute; missing reports, stale identities and bad upstream states fail.
Never quarantine a reproduced unconfirmed refund, tenant leak, or deterministic regression.
Only the named on-call release incident commander may use the restricted ruleset bypass role.
Require an incident ticket, second reviewer, expiry, and ruleset audit record; review the bypass afterward.
Fork approval starts execution but does not supply secrets. Review changes and re-push them to an
in-repository branch. Never use pull_request_target to check out privileged untrusted code.
