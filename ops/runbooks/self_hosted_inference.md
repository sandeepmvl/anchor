# Self-hosted inference

## Required offline evidence

Run the adapter tests, the inherited Rails chat quota/isolation regressions,
and the CPU experiment from the lesson. Default transport is offline.
MockTransport is an in-process protocol server, not vLLM or a quality judge.
The NumPy rounding experiment is not an implementation of AWQ calibration.
No weights, GPU, network, container or hosted provider is exercised.

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

## Release gate

The lesson adapter refuses production construction and release without a
separately implemented, reviewed real-evidence validator. Offline success
never removes that refusal. Keep artifact approval, GPU startup, tenant quota,
isolation, groundedness and live latency evidence before implementing promotion.
For CPU-only GGUF serving choose llama.cpp's llama-server, not an AWQ rename.
TGI is in maintenance mode and is not the default for this new deployment.
