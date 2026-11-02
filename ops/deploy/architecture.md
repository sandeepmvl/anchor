# Anchor deployment decision

## Selected boundaries

Rails accepts authenticated tenant requests and persists durable AI tasks.
Solid Queue workers execute async ticket handling and document ingestion.
The independently deployed AI runtime is the centralized application gateway.
It retains Anchor's authorization integration, inference guards and contracts.
Only Rails exposes a host port. Service authentication is still required.

## Alternatives

- Embedded calls minimize components but couple AI handling to web capacity.
- A queue frees request handlers for asynchronous work; it does not add inference capacity.
- A per-web sidecar shares a replica lifecycle with Rails and couples sizing.
- A centralized gateway pools AI handling and scales independently of Rails.

Three AI replicas and one Rails replica are valid here. Three gateway replicas
are not three model servers and do not increase a hosted provider's quota.

## Provider mix

Hybrid means choosing hosted APIs or self-hosted inference behind the gateway.
It is not a fifth deployment topology. This rehearsal releases only `fake`,
which uses Anchor's existing offline provider. Hosted and self-hosted routes
remain disabled until their established release gates provide live evidence.
An off-the-shelf proxy can replace generic provider plumbing, not ticket policy.

## Single-host persistence

The database service initializes and checks SQLite on a shared named volume.
SQLite clients open that file directly; this is not a database server.
Keep this deployment on one host. A multi-host release requires a server database,
its supported Rails adapter, migrations, backups and separate readiness evidence.

## Evidence

Offline tests prove manifest policy, authenticated fake inference and worker wiring.
They do not prove background-task completion, image builds, DNS, container isolation,
TLS or replica capacity. The optional Docker demo must complete the existing fixture
through Rails and the worker before deployed task completion can be claimed.
