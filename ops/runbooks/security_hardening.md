# Security hardening milestone

## Local acceptance

Run the full hardened-release target through bin/redteam_harness.rb.
It runs the existing security integration suites and writes
.git/anchor-eval/security.json only after success.
Run its negative control to observe rejection of one high-severity finding.
Do not edit production configuration for this exercise.

Run bin/release_gate for the entire accumulated Rails, Python, retrieval,
trajectory, golden and judge release decision on an integration machine.
Its --smoke mode is not a release decision.

## Production evidence

ANCHOR_SECURITY_EVIDENCE names a trusted JSON file accessible to the final
CI job. A path on a different runner is not accessible automatically.
Provision it through the trusted integration runner or a verified artifact
transport. Do not accept evidence from an untrusted pull-request upload.

Use the schema implemented by labs.security_hardening.milestone.security_errors.
Bind evidence to eval.release_gate.manifest(), the full hardened-release
suite, current timestamps, zero high-severity findings, all handler guards,
passing benign controls, the rejected negative control, accumulated suites,
and reviewable evidence links. Record production scope and true results for
deployed_http_auth, live_detector_evaluation, payment_retry_recovery and
sandbox_isolation only after those checks really run.

## Measurement and rollback

Attach per-boundary paired timings, provider usage and prices, benign failure
counts, suite logs, revision and risk-owner review to the audit.
The offline microbenchmark covers only the shared input boundary.
On a failed invariant, stop promotion. Roll back using the existing release
procedure; retain sanitized evidence and revoke affected credentials if needed.
