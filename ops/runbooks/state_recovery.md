# Anchor state recovery

## Local exercise

The shell entry points live in ops/backups. The fixture uses Anchor's SQLite
knowledge schema plus explicitly named recovery_* projection tables. It adds a
checksum column only to its disposable fixture, not to the production schema.
No model calls, external refunds, Rails requests, or production backups occur.

1. Supply an ephemeral 32-byte hex ANCHOR_BACKUP_KEY. Keep it outside snapshots.
2. Run the offline snapshot entry point. Inspect the encrypted path and manifest.
3. Run the drill entry point. Only tmp/anchor-recovery/disposable.sqlite3 is allowed.
4. Compare all table counts and record checksums. Inspect the chunk checksum column.
5. Record snapshot age, elapsed_seconds, rto_met, and the limitations in the report.
6. Remove the local key and tmp/anchor-recovery after recording exercise evidence.

## Production: blocked until a coordinated adapter exists

Disable ingestion, memory writers, and tool workers using the incident playbook.
Inventory every database, native claim store, attachment, index, prompt, config,
and golden dataset. Coordinate their immutable versions at one recovery boundary.
Restore into isolation; verify counts, checksums, ownership, and release compatibility.
Reconcile provider action IDs and original idempotency keys, including the recovery gap.
Test encrypted off-host recovery and key availability. Measure RPO and usable-service RTO.
Only an approved production adapter and reconciliation evidence can authorize promotion.
