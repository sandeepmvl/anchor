# Anchor on-call checklist

Copy this template into the incident record; never include credentials.

- [ ] Shift start/end, named primary and secondary recorded.
- [ ] Escalation contact and incident commander route confirmed.
- [ ] Incoming operator tested pager receipt and dashboard/worker access.
- [ ] Open alerts, recent releases, maintenance and hypotheses reviewed together.
- [ ] Every active issue has evidence, next action and a deadline.
- [ ] Incoming operator explicitly accepted each issue with a timestamp.
- [ ] Outgoing operator remained responsible until acceptance.
- [ ] Repeated pages and manual toil filed for engineering review.

## Incident record

Record incident ID, detected_at, actor, queue/tenant scope, alert samples,
trace references, action time, command scope, result and remaining risk.
Attach recovery.json and actions.jsonl for the fixture. Record the accepted
next owner and next update time. Command success alone does not close a page.

## Maintenance review

For every entry in ops/maintenance/schedule.yml record a named person,
last completion evidence, next due date and explicit acceptance of ownership.
For an exercise, name the learner primary and mark an unstaffed secondary,
untested paging and missing production contacts as production readiness gaps.
