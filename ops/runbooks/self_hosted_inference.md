# Self-hosted inference

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
