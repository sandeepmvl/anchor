# Anchor ticket surge benchmark

Status: not run.

Run ops/load/ticket_spike.py to populate measured offline results.
No throughput, latency, or saturation measurement is claimed here.

Required evidence:
- Successful-ticket p50, p95, and p99; useful completions per second.
- Worker queue wait separate from model-call elapsed time.
- Admission, provider 429, timeout, and transport failure counts.
- Batched endpoint results and scheduler waiting.
- Per-tenant admission outcomes and the first sampled knee rule.

Decision: production capacity changes are not approved without live evidence.
