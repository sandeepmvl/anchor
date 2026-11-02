# Staff design review

## Decision and permanent contract
Empty retrieval evidence or exhausted execution budgets require a saved human escalation.
No autonomous tool follows that decision, including duplicate starts and checkpoint resume.
Approval checkpoints remain distinct: a scoped reviewer may approve the exact pending action.
Evidence: anchor_ai/tests/test_bounded_runtime.py and test/agents/bounded_agent_test.rb.

## Architecture defense
Evidence: ops/deploy/architecture.md; bounded and multi-agent routes in anchor_ai/app.py.
Decision: authenticated identity and tool authorization remain outside generated text.
Pending review: cite exact routes, isolation results, fan-out limits, and scaling observations.

## Incident decision
The planted fault skips the terminal transition after empty evidence.
Containment: stop autonomous actions while preserving intake and audit history.
Prefer a compatible known-good rollback for a code-only fault; inspect state compatibility first.
Pending review: attach trace ID, first divergent event, release comparison, and stranded-ticket reconciliation plan.

## Cost and platform decision
Pending review: cite the existing comparison matrix path, input period, and generated rows.
Record daily tokens, token mix, hosted marginal rate, local fixed and variable costs, and crossover calculation.
Record quality, burst latency, recovery, retention, and tenant-specific failover qualification.
Do not substitute the lesson's fictional prices or latency assumptions for Anchor evidence.

## Release boundary
Offline tests verify deterministic runtime contracts, not deployed authentication or live serving.
Production release remains blocked without the existing container and live-provider evidence.
