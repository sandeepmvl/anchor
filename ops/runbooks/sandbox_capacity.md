# Sandbox capacity and release gate

## Current release
The process backend is an explicit local rehearsal. It runs supplied fixtures only.
Production requests return isolation_unavailable. Never silently fall back to it.
The Dockerfile, runtime policy, and seccomp profile are deployment inputs, not a
working container launcher or evidence that isolation was installed.

## Admission
The rehearsal uses a lock around service-local global and per-tenant active counts.
It charges one run and one cost unit before launch and releases active capacity
in finally. Its run and cost budgets last for that service instance; they are not
persistent billing or a distributed rate limiter. Run one rehearsal service.

Before production, implement shared admission across all workers with atomic
leases, bounded queue length, tenant rate windows, persistent cost reservations,
lease expiry, crash reconciliation, and a separate container-reaping supervisor.
Reject when that coordinator, required isolation, or clean pool capacity is absent.
Cancellation must not release a production lease until the container is gone.
Do not use an asyncio cancellation as evidence that a worker was terminated.

## Isolation release gate
Pin and preload the reviewed image by digest. Validate the supplied seccomp
allowlist against the target image and architecture. Verify effective CPU, memory,
pids, read-only root, non-root identity, dropped capabilities, no-new-privileges,
no host mounts or sockets, bounded scratch, and network=none before accepting jobs.
A failed check prevents dispatch. Never switch to privileged or unconfined mode.
An external supervisor must kill and remove the whole container at the deadline,
even if the requesting application exits. Bound startup, execution, and output.

Record real socket failure with namespace settings. Record fork EAGAIN with an
increase in pids.events max. Collect counters before teardown. An exit status of
137 alone does not prove OOM; collect OOM state and memory.events separately.
JSON parsing and Dockerfile assertions do not satisfy this release gate.

## Egress
Current diagnostics need no network. Keep network=none. The tool broker's exact
HTTPS hostname gate is not a raw-socket firewall. Its local transport is absent.
If a proxy is introduced, require a proxy-only namespace, default-drop IPv4 and
IPv6 rules, private/metadata IP rejection after DNS, no redirects, authenticated
broker authorization, exact destination policy, and a final deny rule. Test direct
IP bypass independently. An allowed domain is not permission to upload tenant data.

## Capacity review
Review measurements around 50 concurrent executions. This is a trigger to measure,
not an established throughput threshold. Separate queue wait, image allocation,
startup, execution, output bytes, denial rate, peak memory, and cleanup latency.
Reject or bound the queue instead of creating unlimited containers. If startup
cost dominates, prewarm clean single-use sandboxes and replenish after each job.
Never reuse writable tenant state or prewarm with credentials or tenant exports.

## Audit and incident response
Correlate tenant, conversation, trace, backend, exit reason, and output byte count.
Do not log scripts, environment variables, raw output, or customer exports.
On isolation drift or unexplained egress, disable admission and destroy suspect
instances. Preserve configuration and resource events before cleanup where safe.
Existing financial tools retain confirmation, durable idempotency, and approval.
The diagnostic entry point exposes only the approved read-only script identifier.
