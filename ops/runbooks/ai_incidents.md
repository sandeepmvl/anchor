# Anchor AI incident response

## Scope and roles
Use the highest matching severity. Detection starts all clocks.
Commander owns priorities; operations changes controls; communications sends updates.
Record actual names and acknowledgment times in an independent incident document.
A delivered page is not acknowledgment. An incoming commander explicitly accepts command.
Security incidents also follow the security response process.

## Severity and routing
| Severity | Trigger | Acknowledge | Contain | Route / updates |
|---|---|---|---|---|
| 1 | Cost rate >=10x approved baseline; confirmed harmful or widespread unsupported answers | 5 min | Immediately; verify human banner within 60 sec | Primary + backup now; updates every 15 min |
| 2 | Cost rate >=3x and <10x; sustained quality/provider failures blocking some tickets | 15 min | 30 min | Primary; backup at 15 min without acknowledgment; updates every 30 min |
| 3 | Isolated degradation with a safe workaround and no higher criterion | 4 hours | 1 business day | Named owner; updates daily |

A failed automated quality score alone does not establish confirmed harm.
The cost baseline must be approved and use the same units and window as the signal.
The offline ledger measures reserved cost bounds, not billed currency.

## Detect and classify
Preserve detection time, window, approved baseline, tenant scope, trace IDs,
reservation IDs, attempts, release versions, and uncertainty. Never copy secrets.
Record the severity and incident ID in AiOperatingMode.audit using declare!.
Its alert field records routing intent; the on-call adapter must deliver the pages.
Reclassification never resets the original detection clock.
For unknown scope, identify every affected tenant and apply containment to each;
a single tenant switch is not global containment. Escalate if scope cannot be bounded.

## Contain and verify
Operations uses an authenticated persisted admin Membership for the affected tenant.
Production service call from the authorized operational console:
`AiIncidentControl.contain_incident!(member: member, organization_id: tenant_id, incident_id: incident_id)`
Do not find an arbitrary admin to impersonate an operator.
Preserve the resulting actor_id, reason, timestamp, and revision in the audit.
Request the same ticket. Verify the configured human message and full sanitized context:
message, history, evidence, proposed action, ticket ID, and idempotency key.
Measure detection-to-banner time, not time since the containment command.
Check workers for zero new model/tool dispatch. Inspect in-flight and uncertain attempts
separately: a local stop acknowledgment does not cancel a sent provider request.
Do not delete reservations, reset quota counters, or replay uncertain actions.
If the deadline is missed, keep containment, escalate, and record the miss.

## Isolated rehearsal commands
Run from the Anchor repository root, in this order:
`bash ops/drills/simulate_cost_spike.sh simulate`
`bash ops/drills/simulate_cost_spike.sh classify`
`bash ops/drills/simulate_cost_spike.sh contain`
`bash ops/drills/simulate_cost_spike.sh chat`
`bash ops/drills/simulate_cost_spike.sh recover`
The adapter opens only var/incident_playbook/rehearsal.sqlite3.
Simulate refuses to overwrite an existing run. Archive evidence before a new drill.
Pages are recorded, not delivered. The service handoff is not an HTTP/browser test.

## Recovery checklist
Keep human-only active until all checks pass over at least 300 seconds:
- Quality: representative reviewed answers and regression gates pass.
- Budget: reconciled spend and bounded retry/admission tests pass.
- Backlog: inspect workers, queues, pending actions, and cancellation outcomes.
- Reconcile uncertain usage and payment effects before retrying anything.
Use recover_incident! with explicit check results and authenticated operator authority.
Production also requires all config/incident_control.yml production_evidence entries.
Missing evidence blocks recovery. Never use the low-level change! call to bypass review.
The required rehearsal demonstrates refusal, not production readiness.

## Postmortem template
Required for SEV1/2, manual containment, missed targets, or monitoring failure.
- Incident ID / severity / commander / operations / communications:
- Impact: window, scope, excess spend, human queue impact, unknowns:
- Timeline: detection, acknowledgment, escalation, activation, banner, recovery, closure:
- Redacted evidence references and review status:
- Contributing conditions: observed facts separately from hypotheses:
- Response: what worked, delays, missing evidence:
- Actions: named owner | due date | priority | measurable completion test:
- Reviewer | review date | distribution | follow-up tracker:
Containment is not repair. Closing the drill does not authorize reopening autonomy.
