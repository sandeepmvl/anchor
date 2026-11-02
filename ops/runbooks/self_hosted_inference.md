# Self-hosted inference

## Versioned release milestone

The required entry points are anchor_ai/tests/test_self_hosted_release.py,
test/integration/self_hosted_failover_test.rb, and
ops/deploy/self_hosted_release.py. The Ruby test is a standalone boundary
contract, not a booted Rails endpoint test. The harness uses a fixture ledger;
production retains RoutingProvider.physical and its verified Rails quota scope.
No tool executor runs in this generation-only harness. Existing confirmation,
dry-run, reversal, authorization, quota, and trajectory tests remain required.

ServingProfiles snapshots immutable profiles and atomically replaces a single-host
pointer. New requests pin the new pointer; in-flight requests finish with their
original snapshot. Select previous to roll back model, prompt, index and routing
versions without changing app code. Never replay committed tool actions. Unknown
tool outcomes require reconciliation, not a new model's guess. Multi-host pointer
distribution requires a separately reviewed transactional control plane.

Use the reviewed matrix as a recommendation only. The release builder opts in
chat and leaves unsupported workloads ineligible. Intersect approved providers,
region, evaluated model, readiness, workload and external-processing permission.
Missing permission means human escalation. A cache key includes the effective
policy, tenant, conversation, profile and committed context; incompatible changes
cannot reuse an earlier answer. Old entries need normal bounded cache eviction
before this illustrative in-memory cache is used in a long-lived service.

For startup, validate ops/deploy/vllm_service.yml, approved immutable artifacts,
model alias and private endpoint before enabling the cohort. Existing defaults
remain hosted. Probe readiness and synthetic generation separately. Schedule
ResilienceBoundary.health_check_once with a bounded operator-owned scheduler;
never start background tasks at import. Probes consume their own accounted
physical sends. Ordinary requests never bypass an open circuit. Offline release
profiles use two total attempts, no per-provider retry, a shared deadline,
one-second cooldown and two successful recovery probes. These are fixture settings.

For scaling, warm and verify a new replica before admission. Confirm model and
KV metric freshness, memory headroom, queue delay and hosted fallback capacity.
Drain bounded in-flight requests before retiring the old replica. A process
restart alone does not clear circuit state. After crash or stall repair, require
cooldown and consecutive successful probes before returning local traffic.

The emergency kill switch is ReleaseService.human_only, set only by trusted
operator control. It bypasses cache and models and returns redacted messages,
evidence, plan position, remaining budget, trace and committed action references.
Bind this control to authenticated operations configuration before production.
Do not expose it as a customer request parameter. Human-only prohibits new
side-effecting actions; pre-existing committed actions remain visible as context.

Monitor selected versus served profile/provider, circuit skips, readiness,
recovery probes, deadline and timeout counts, handoff count, queue delay, KV
pressure, hosted quota, latency and error rate. Correlate spans using trace IDs;
do not log prompt bodies. Page inference on-call for local failures or readiness/
generation disagreement; page Anchor application on-call for failed ticket
completion or exhausted fallback quota. Escalate dual failure to the support
incident commander and force human-only mode. Bind these roles to the paging
directory and configure SLO-based alert thresholds before production approval.
Missing role bindings or thresholds block operational handoff.

The accumulated release gate still requires unit, retrieval, trajectory, golden
quality and calibrated judge evidence. Set ANCHOR_SELF_HOSTED_RELEASE=1 for a
self-hosted production candidate. Missing real infrastructure evidence remains
fail-closed even if every offline test passes. The offline gate command captures
and prints this expected refusal; it is not a successful deployment.

## Required offline evidence

Run the adapter tests, the inherited Rails chat quota/isolation regressions,
and the CPU experiment from the lesson. Default transport is offline.
MockTransport is an in-process protocol server, not vLLM or a quality judge.
The NumPy rounding experiment is not an implementation of AWQ calibration.
No weights, GPU, network, container or hosted provider is exercised.

## Surge batching profiles

The versioned batching_profiles in config/self_hosted.yml are separate from
the inherited single-request benchmark. The deployment specification commits
the default profile. ops/load/load_test.py validates that command and renders
candidate arguments for the selected profile without editing or deploying it.
The tuned candidate uses 16 sequences and 512 scheduled tokens per iteration.
A reviewed production change must update the selected profile, deployment
profile name, and matching command arguments together; drift fails validation.

Run the same concurrency and scheduling mode for both profiles. Record completed
requests, rejected admissions, timeouts, queue wait, completion p99, short-ticket
first-token p99, and each tenant's counters. A failed request is never successful
throughput. The in-process OpenAI-compatible server supplies synthetic lengths;
the scheduler uses virtual time and an explicit assumed iteration cost.
Full-context KV estimates exclude runtime overhead and do not prove device fit.
Tenant token reservations model conservative cost quotas, not actual billing.
Keep Rails tenant authorization, rate limits, and physical-attempt quotas enabled.

Offline reports cannot promote a profile. require_surge_promotion refuses even
claimed live evidence until a separate reviewed real-evidence validator exists.
The offline tests prove that refusal. No weights, quantization quality, GPU
memory use, live cancellation behavior, or real throughput is verified here.

## Configuration

Give Rails and Python LLM_BACKEND=self_hosted and
ANCHOR_VLLM_TRANSPORT=offline for protocol testing. Rails retains ANCHOR_AI_URL
and its service secret. LLM_BASE_URL belongs to Python; it is not the Rails
AI-service URL. Never accept a destination or backend from customer parameters.
Hosted rollback uses LLM_BACKEND=hosted only for tenants authorized for hosted
processing. Residency-constrained tenants fail to human review instead.

For the existing multi-provider router, add an operator-owned deployment with
backend: self_hosted, model: tiny-llama and the correct private region. Add it
to the order and explicitly approve it for the test tenant. Do not approve
other tenants or set evaluated true based on protocol fixtures. The router's
existing quota scope, physical-attempt reservation and settlement stay required.
Buffered routing still rejects tools and streams; direct adapter compatibility
does not expand the router's capabilities.

## On a GPU machine

Install the pinned runtime from the deployment specification in an approved
GPU environment. Validate the approved artifact manifest, immutable digest,
Llama architecture, AWQ packing, tokenizer, chat template and GPU capability.
Run the specification's command, then check /v1/models for tiny-llama.
Use a private authenticated TLS proxy when crossing hosts. Configure
ANCHOR_VLLM_TRANSPORT=live, LLM_BASE_URL and LLM_API_KEY in the Python service.
Submit the same authenticated tenant ticket through Anchor, with quota
admission enabled. Capture redacted retrieval, tool and LLM spans, actual usage,
answer grounding and warm/cold latency for both authorized destinations.
Record failures too. Equal numeric token caps do not imply equal model work.
For batching comparisons, launch each validated profile on the same GPU and
approved artifact. Use the same redacted ticket distribution, concurrency,
warmup, deadlines, tenant admission, and measurement duration. Capture actual
input/output tokens, completed throughput, timeouts, queue wait, first-token
latency, completion p99, and tenant-level outcomes. Compare one setting at a
time after the initial two-profile experiment. Attach the immutable profile,
artifact digest, runtime version, GPU identity, and raw measurements to review.
The offline surge runner is not a live load generator; use the established live
adapter and an approved authenticated load client for these measurements.

## KV pressure and bounded admissions

The product module anchor_ai/kv_cache.py budgets per tensor-parallel rank.
Use num_key_value_heads, cache dtype, block-rounded prompt-plus-output lengths,
resident weights, and non-KV overhead. AWQ weight packing does not imply an
AWQ KV dtype. These assumptions cover full attention without prefix-sharing
credit; confirm sharding and replication against the deployed architecture.

The adapter owns one Admission instance per backend instance. Deploy one shared
backend/controller per replica, not a new instance per ticket. Each physical
send reserves a full context budget and releases it on completion, timeout or
cancellation. Excess requests receive a 503 with a bounded error code; this
controller has no queue and no retry loop. Existing tenant quotas and the
caller's finite retry/deadline policy remain required. A context-length limit
must be enforced by the server/tokenizer; never silently truncate tickets.

Offline transport supplies healthy metric fixtures, not GPU observations.
Live transport starts with no sample and rejects admissions. Before allowing
live traffic, implement and review an operator-owned collector with a timeout,
verified HELP names and units, and per-replica aggregation. Feed normalized
samples to the shared kv_admission.update method using the controller's
monotonic clock for receipt timestamps. Missing, stale or future samples fail
closed. Do not let customer payloads update this controller.

Use ops/load/kv_cache_dashboard.py for normalized offline fixtures and finite
virtual-time experiments. Use load_test.py --kv-profile for the same surge
entry point. Unsafe mode bypasses admission only in that experiment. Recovery
costs and virtual throughput are assumptions; measured_cpu_ms is real Python
elapsed time, not GPU latency. No swap or CUDA OOM is verified. Admission spans
inherit the active request trace and contain no ticket text or tool arguments.

On a GPU machine, validate the existing approved launch command and compare
startup-reported KV blocks with the estimate. Check the pinned backend supports
16/32-token blocks before tuning. Replay identical authenticated long-context
tickets with quotas and deadlines. Correlate occupancy, waiting, counter deltas,
completed throughput and real latency. Record real failures and cancellation
behavior. A rising preemption counter is not an allocation traceback. Keep the
production gate closed until a reviewed real-evidence validator exists.

## Release gate

The lesson adapter refuses production construction and release without a
separately implemented, reviewed real-evidence validator. Offline success
never removes that refusal. Keep artifact approval, GPU startup, tenant quota,
isolation, groundedness and live latency evidence before implementing promotion.
For CPU-only GGUF serving choose llama.cpp's llama-server, not an AWQ rename.
TGI is in maintenance mode and is not the default for this new deployment.

## Workload hosting economics

The cost dashboard's comparison export is restricted to the selected organization's
administrators. It contains daily allowlisted workload aggregates, not ticket text,
customer identifiers or raw usage records. Existing usage events must already pass
Anchor's redaction-before-storage policy; this export does not repair unsafe source
storage. Unknown feature names become `other`, never arbitrary exported strings.
Unknown workload policies stop the Python comparison until an operator reviews them.
Embedding tokens are added to input demand for the kb_embedding workload; verify
that the source accounting convention does not count them in input_tokens too.
Cost fields retain the dashboard accounting unit. Missing cost or latency stays
missing. Event latency percentiles have unknown warm/cold state and do not establish
startup, recovery or end-to-end SLO compliance.

bin/compare_hosted_local.py reads the committed 30-day fixture by default, or a
JSON Lines export through --usage. It writes a matrix with conditional choices,
fully loaded cost, idle capacity, reserved headroom and break-even sensitivity.
The comparison allocates a separate fleet per workload; do not sum these estimates
as a shared-fleet bill. Utilization is a planning demand/capacity assumption, not
measured GPU occupancy. Replica counts are rounded up. Fixed-replica crossings
must also fit the reported usable capacity; adding replicas creates cost steps.
External processing approval and quality equivalence are assumptions in the offline
policies, not tenant authorizations. No report can change production routing.

### Required offline economics checks

Run the named Python economics tests and the standalone Ruby exporter tests from
the lesson. These execute the real policy and aggregation code with offline stores.
They do not verify Rails session middleware, a deployed database schema, GPU fit,
model quality, provider prices, throughput, cold starts or live network latency.
Keep the inherited Rails dashboard authorization regressions in the normal app CI.
The production routing function refuses all reports until a separately reviewed
real-evidence validator is implemented; changing a report's status cannot release it.

### On your machine: replace assumptions with evidence

In an approved environment, export an authorized tenant's month from the dashboard.
Confirm source token conventions, workload mappings, retries and accounting units.
Obtain dated API and managed-endpoint quotes, billing minimums, warm-capacity terms,
GPU region/replica prices, storage/network costs and an on-call operations allowance.
Use equivalent quality-approved models and identical redacted request distributions.
Measure warm end-to-end p50/p99, cold readiness, recovery, failures and peak arrivals
for all three options. Record model/artifact digest, runtime, hardware, region,
concurrency, measurement window and raw evidence provenance. Validate processing
location, retention, access and tenant isolation. Review capacity and privacy before
price. Attach this evidence to review; neither this CLI nor a successful offline test
implements production promotion.
