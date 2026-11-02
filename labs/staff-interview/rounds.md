# Staff interview rounds

## System design — 10 minutes
Open ops/deploy/architecture.md and identify the authenticated Rails-to-Python path.
Find the bounded-agent and multi-agent orchestrator routes in anchor_ai/app.py.
Explain identity verification, scoped retrieval, admission, checkpoints, and external tool authorization.
Use isolation negatives, retrieval evaluation, and trajectory results; state their coverage gaps.
Pushback: what prevents delegation or a retry from widening authority?

## Implementation — 20 minutes
Inspect the planted failure and trace events. Locate the missing saved terminal transition.
Repair the existing runtime and escalation path, not the test expectation.
Explain why awaiting_approval remains resumable but exhausted evidence does not.

## Incident — 10 minutes
Use the failing trace, diff, ops/runbooks/ai_incidents.md, and release history.
Contain autonomous actions first. Identify pending versus committed actions.
Choose compatible rollback or fix-forward; state how stranded tickets are recovered.

## Cost and platform — 10 minutes
Locate Anchor's existing comparison matrix and its generating comparison/exporter code.
Quote the input period, token mix, daily volume, fixed cost, variable rates, and exclusions.
Calculate crossover from those inputs; do not reuse the teaching example as project data.
Defend hosted versus self-hosted and provider failover using quality, latency, retention, authorization, and recovery gates.
Record missing measurements as missing; never label fixture results as production evidence.
