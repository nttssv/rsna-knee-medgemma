# Qwen provisioning runtime v1

Status: **live components implemented; all live switches false; no provider mutations performed**.

This stage implements the transport and outer shutdown controls for the [reviewed provisioning proposal](../qwen_execution_provisioning_v1/README.md). The frozen Qwen experiment, prepared five reports, prompts, parsers, model recipe, execution plan and 40/18 split are unchanged.

| Component | Implementation |
|---|---|
| [Lifecycle controller](scripts/controller.py) | One create, immediate stop, one possible later resume; immutable outer clock and cumulative budget |
| [RunPod transport](scripts/provider_transport.py) | Exact REST request, independent uncached reads, no create/resume retries, private credentials, bounded sanitized responses |
| [Hard call supervisor](scripts/network_supervisor.py) | Separate process group per call; wall-clock plus monotonic deadline; group kill and child reap |
| [External shutdown worker](scripts/external_shutdown.py) | Detached process, OS identity and fresh challenge verification, cached creation ownership, exact-name reconciliation, bounded stop attempts |
| [Live policy and once ledger](scripts/live_policy.py) | Disabled production gates, source manifest verification, exclusive durable operation claims across processes/output directories |
| [Inner handoff](scripts/handoff.py) | Calls the existing frozen inner supervisor and original runpodctl preflight verifier; cannot substitute the external worker for the inner watchdog |

The resource boundary stays Secure Cloud, one NVIDIA A40 48 GB in CA-MTL-1, 80 GB container disk, zero persistent volume and no network volume. Compute is capped at $0.49/hour and storage at $0.012/hour. Total cost remains at most $1.50, within one outer window of at most 3,600 seconds; the stop reserve begins exactly 300 seconds before its deadline. The clock includes the initial read and guard setup before the sole create request. No replacement pod, GPU, region or physical machine is allowed.

Production construction rejects disabled policy before credentials or network access. There is no automatic approval writer, remote bootstrap, source-enable shortcut, fallback or controller recovery. The original session object must remain alive for its one later resume; allocator death is terminal and activates external cleanup. Durable intent-level claims prevent replay through a new output directory.

```bash
.venv/bin/python -m pytest -q analysis/qwen_execution_provisioning_runtime_v1/tests
.venv/bin/python analysis/qwen_execution_provisioning_runtime_v1/scripts/controller.py --execute
```

The second command intentionally reports disabled execution, exits 2, and makes no provider calls. Tests use fabricated receipts, local subprocesses and loopback HTTP only. They are not GPU, billing, live-stop or model results.

The documented REST readback does not establish actual stopped state, current zero billing, separate compute/storage rates or GPU VRAM. Unknown fields remain unverified. The production factory rejects this known observation limitation **before creation**, even if its policy were later enabled. Adding a trustworthy observation source is a concrete prerequisite, not permission to infer missing values.

The handoff is a co-located adapter for the existing inner launcher. It does not upload files, configure SSH, prepare cache or deploy the original pod-local watchdog. Those resources must be available on the exact approved pod; the frozen inner switch remains false. See the [protocol and blockers](PROTOCOL.md), [validation results](RESULTS.md), [source hashes](configs/source_manifest.json), and [pinned image provenance](IMAGE_PROVENANCE.md).
