# Anchor operations runbook

## Ownership and entry

Service owner: Anchor platform team. Resolve named primary, secondary and
incident commander through on_call_checklist.md before production work.
Dashboard: anchor-tickets, panel QueueAgeHigh — worker recovery.
Related safety procedure: ops/runbooks/ai_incidents.md.

## QueueAgeHigh catalog

The fixture alerts when oldest pending age is at least 2000 ms and completed
count is unchanged across two samples at least one second apart.
Pending age is a latency lower bound, not a completed-request percentile.
Production paging requires a deployed queue datasource and alert rule; the
text dashboard panel and fixture do not deliver pages.

1. Record alert time, actor, affected queue, tenant scope and recent changes.
2. Check oldest pending age and completed-job progress, not Rails health alone.
3. Compare queue wait with retrieval, tool and LLM span durations. The fixture
   provider takes about 10 ms. Its report is not a production provider trace.
4. Long provider spans with progressing completions support provider triage,
   not a worker restart. Check errors, retries and the provider runbook.
5. If queue and provider are healthy, check semantic-cache hit rate and recent
   changes. Missing evidence means unknown. Escalate an unmatched diagnosis.
6. Suspected tenant leakage or unauthorized actions: invoke ai_incidents.md,
   use its human-only control, and stop this routine repair.

## Offline bounded action

Run `ruby labs/production-operations/worker_recovery.rb inspect` and confirm QueueAgeHigh.
Run `ruby labs/production-operations/worker_recovery.rb restart` once. It starts a stalled
fixture child, sends TERM, and waits for that child to drain its durable queue.
Run `ruby labs/production-operations/worker_recovery.rb verify` to submit a fresh 20-ticket window.
Do not reset a failed attempt before saving its evidence.

## Production bounded action — not authorized by fixture evidence

Resolve the dedicated worker supervisor PID and shutdown timeout with the
platform owner. Confirm its identity, scope, claim state and tool reconciliation.
The graceful shutdown command is `kill -TERM "$ANCHOR_WORKER_SUPERVISOR_PID"`.
Wait for supervisor exit under the approved timeout, then start the same worker
through its deployment supervisor. Never target the web process or bulk-requeue
claims. If identity, timeout, reconciliation or progress is unknown, stop and
escalate. Do not use KILL as a routine recovery step.
The drill's production gate refuses authorization because it has no deployed adapter.

## Verification and exit

Within 300 seconds of detection, require backlog drained, preserved ticket and
action keys, and a fresh 20-request p95 below 2000 ms. Keep the original backlog
p95, window boundaries and sample counts. Production also requires actual
customer delivery, Solid Queue claims and external tool reconciliation evidence.
Record actor, evidence, action, result, remaining risk and accepted next owner.
Escalate at the deadline or immediately if the action is unsafe or unexplained.
