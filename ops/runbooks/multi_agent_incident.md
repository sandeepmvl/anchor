# Multi-agent incident

## Stop automation

Use the same durable orchestration database for every worker. A worker-local
SQLite file is not a shared kill switch. Multiple hosts require a supported
shared database service or a single operational-control owner.

Run `python3 -m ops.deploy.orchestration_release --database PATH --operator INCIDENT stop`.
Record the incident, operator, timestamp, and active version.
Workers must reread the mode before model work and immediately before tool dispatch.
Save the redacted handoff before acknowledging stopped work.
Transfer ticket, evidence, completed actions, pending actions, approval state,
trace identifier, orchestration version, and reason.

A mode change is not cancellation of an already committed external side effect.
Reconcile receipts and idempotency keys before retrying that work.

## Restore the prior orchestration version

Keep human-only mode active. Both versions must already be registered as
immutable, validated configurations in the operational database.
Run `python3 -m ops.deploy.orchestration_release --database PATH --operator INCIDENT rollback --expected FAILED --target PRIOR`.
A stale expected version rejects the change. The active pointer changes in one
transaction and human-only mode remains active. No application deployment is needed.
Do not edit committed YAML as an incident rollback.
Keep in-flight work pinned to its original version or escalate it.

## Resume

Run `ruby bin/release_gate` against the real dependency stack.
Inspect queue age, tenant quotas, MCP availability, sandbox capacity, and traces.
The offline smoke suite does not establish dependency health or isolation.
Only the incident owner may clear human-only mode after the production gate passes.
