# Anchor security audit

Edition: OWASP LLM Top 10 — 2025.
Decision: PENDING local execution; production promotion BLOCKED without evidence.
Revision / candidate manifest: PENDING.
Reviewer / risk owners / evidence links: PENDING.

## Coverage map

| Category | Evidence to inspect | Residual or uncovered risk |
|---|---|---|
| LLM01 Prompt Injection | Message/document detector tests, handler guards, bounded-runtime tests | Novel, encoded and live-model attacks |
| LLM02 Sensitive Information Disclosure | PII pipeline and tenant integration tests | Detector recall, output disclosure and logs outside tested paths |
| LLM03 Supply Chain | Not established by this milestone | Dependencies, model provenance and build integrity |
| LLM04 Data and Model Poisoning | Hostile retrieved-document cases provide partial evidence | Training poisoning and ingestion provenance |
| LLM05 Improper Output Handling | Existing response validation is partial evidence | Audit every downstream sink separately |
| LLM06 Excessive Agency | Exact confirmation, current permission, dry-run and reversal suites | External payment recovery and deployed privileges |
| LLM07 System Prompt Leakage | Not established by this milestone | Extraction and indirect leakage evaluation |
| LLM08 Vector and Embedding Weaknesses | Existing tenant retrieval/isolation suites provide partial evidence | Deployed index ACLs and ranking attacks |
| LLM09 Misinformation | Accumulated golden and judge suites | Judge calibration, distribution shift and unsupported answers |
| LLM10 Unbounded Consumption | Existing quota and bounded-agent suites provide partial evidence | Distributed exhaustion and deployed capacity |

## Executed results

Security-focused offline harness: PENDING.
High-severity findings / total executed cases: PENDING.
Benign failures / total controls: PENDING.
Known-regression promotion rejection: PENDING.
Entire accumulated release gate: PENDING; do not infer it from the security harness.

## Latency and cost

Attach .git/anchor-eval/security.json for paired shared-input-boundary timings.
Record added provider calls and cost separately from CPU/infrastructure cost.
Full stacked production overhead, representative false-positive impact and
infrastructure evidence remain PENDING until measured on the integration system.
No offline result authorizes production deployment.
