# Qwen provisioning lifecycle: local runtime v1

Status: **local synthetic controller; live execution unavailable**. This version tests the operational sequence proposed in [provisioning v1](../qwen_execution_provisioning_v1/README.md). It cannot provision or resume a real pod, creates no real approval, and contains no model-cache, download, training, or inference entry point.

The existing inference runner requires a stopped-pod observation and control preflight. A newly created RunPod pod begins running immediately. The controller therefore separates one initial create-and-stop phase from one possible later resume, under a single clock and cumulative budget. The reviewed inner runner is imported only for its existing grant/observation and source-integrity checks; none of its files changes.

| Boundary | Value |
|---|---|
| Exact allocation | Secure Cloud, CA-MTL-1, one NVIDIA A40, 48 GB |
| Storage | 80 GB container disk, zero persistent volume, no network volume |
| Price ceilings | Compute $0.49/hour; storage $0.012/hour |
| Cumulative maximum | $1.50 across initial creation and any later resume |
| Outer clock | At most 3,600 seconds from the recorded first create intent; never reset |
| External and eventual pod stop time | Outer deadline minus exactly 300 seconds |
| Creation / later resume | At most one request each; no automatic retries or replacement |
| Physical machine | Bound at the first verified read; a changed machine after resume fails |
| Initial command | `/bin/sleep infinity`, pinned image by immutable amd64 manifest digest |

The source records the first-create timestamp before guard readiness verification, which conservatively includes that verification delay in the outer clock. `create-attempt` must be fsynced before dispatch. A lost response triggers at most one exact-name read reconciliation; it never triggers a second create. A unique preread-absent intent name can identify a cleanup target even when its allocation is wrong, but that target never becomes valid for inference.

Stop verification uses a separate observer interface. It requires a fresh exact-pod stopped state and $0/hour. Stop-command success alone does not establish either. Emergency stop attempts continue if local receipt writing fails; the controller records a durability failure and forbids later progress. At most three stop attempts are requested per shutdown phase.

The [implementation](scripts/controller.py) and [protocol](PROTOCOL.md) are explicit about their limit: **injected synchronous adapters are local test interfaces**. No production transport, independently deployed shutdown worker, or hard network-call supervisor is implemented. A `synthetic_only` attribute is a testing contract, not a security sandbox or verified production identity. The CLI unconditionally returns `LIVE_EXECUTION_UNAVAILABLE`, even with `--execute`; changing the documentary JSON flags cannot enable it.

```bash
.venv/bin/python -m pytest -q analysis/qwen_execution_provisioning_runtime_v1/tests
.venv/bin/python analysis/qwen_execution_provisioning_runtime_v1/scripts/controller.py
```

The second command intentionally exits with status 2 and makes no provider calls. The tests fabricate all provider responses, approvals, observations, and guard attestations. They do not demonstrate live availability, real stopping, billing, CUDA behavior, or model performance.

The exact image digest binds its inherited entrypoint too. Overriding CMD with sleep does **not** establish that inherited entrypoint/hooks perform no other work. Their inspection, an exact provider request mapping, and production failure supervision remain required before live use. Mutable provider template settings are not accepted (`template_id = null`). See [image provenance](IMAGE_PROVENANCE.md).

Read [RESULTS.md](RESULTS.md) for the validation status. No scientific artifacts, inner runtime switches, resource proposals, reports, prompts, parsers, token artifacts, or 40/18 split were modified.
