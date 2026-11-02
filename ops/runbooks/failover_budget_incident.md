# Failover budget incident

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
explicit authorized change; this drill deliberately supplies no reset switch.

Offline evidence covers one-host SQLite admission and synthetic dispatch only.
Production remains blocked without authenticated paid-path, multi-host,
kill-switch race, provider cancellation, and calibrated hosted-judge evidence.
