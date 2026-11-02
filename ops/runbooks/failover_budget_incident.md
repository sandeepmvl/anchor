# Failover budget incident

Use [AI incident response](ai_incidents.md) for severity, named ownership,
alert escalation, the 60-second severity-1 human-banner target, and the
postmortem template. Detection starts the deadline; classification does not
reset it. Run the isolated replay with
`bash ops/drills/simulate_cost_spike.sh simulate`; this is not a live page.

## Diagnose
Preserve trace IDs, ticket IDs, window keys, attempt counts, circuit decisions,
remaining deadlines, reservation IDs, and spend/latency series.
Reject at least one rival using evidence: constant tokens rule out token growth;
serial execution rules out parallelism as a necessary cause.
An outage triggers retries. Repeated zero window usage explains amplification.

## Contain
Use the authenticated operational authority to activate tenant human-only mode.
Disable unhealthy routes through the existing authorized breaker override.
Do not reset counters, delete reservations, or retry uncertain provider calls.
Queued and new tickets must enter the human queue with message, history,
evidence, proposed action, and the blocking reason intact.

## Verify and recover
Reject cyclic routes, per-request windows, and multiple retry owners.
Run the identical failure trace, concurrent admission tests, restart tests,
and accumulated Rails, Python, retrieval, trajectory, and judge regression gates.
Verify zero new AI dispatch after human-only activation, not merely lower spend.
Reconcile uncertain usage before changing limits. Resume gradually through an
explicit authorized change through `AiIncidentControl.recover_incident!`.
The incident drill deliberately supplies no successful reset command.
Keep the existing low-level `change!` authority restricted to authenticated
operators; it is not a recovery shortcut. Require the recovery checklist
before any operational reenable, including a direct administrative change.

Offline evidence covers one-host SQLite admission and synthetic dispatch only.
Production remains blocked without authenticated paid-path, multi-host,
kill-switch race, provider cancellation, and calibrated hosted-judge evidence.
