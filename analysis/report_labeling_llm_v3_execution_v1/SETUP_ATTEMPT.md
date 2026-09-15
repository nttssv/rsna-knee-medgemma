# Approved setup attempt · 15 September 2026

The original stopped pod could not restart: RunPod explicitly reported that its GPUs were no longer available. The user then approved one replacement RTX 6000 Ada at the same $2 budget and 90-minute maximum. This explicit instruction superseded the proposal's restriction to the original stopped pod; it did not change model settings or the experiment.

## Observed resources

| Item | Observed value |
|---|---|
| GPU | NVIDIA RTX 6000 Ada Generation, 49,140 MiB |
| CPU / host RAM | 16 vCPU, AMD EPYC 9354; 188 GB RAM |
| Image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` |
| Python / PyTorch | 3.12.3 / 2.8.0+cu128 |
| CUDA / driver | PyTorch CUDA 12.8 / NVIDIA 595.91.07 |
| Native BF16 | Supported |
| Storage | 80 GB temporary container disk; no persistent volume |
| Actual rates | $0.84/hour compute + $0.011/hour container storage |
| Start request / stop verified | 08:00:13 / approximately 08:06:55 UTC |
| Cost | Approximately $0.10 estimated; not an invoice |
| Final provider state | Compute and container storage not running; $0.00/hour |

The allocation supplied more CPU/RAM than the catalog minimum without increasing the approved GPU/disk rates. These are setup observations, not model runtime measurements. The remaining nine pinned Python dependencies were not installed because the shutdown prerequisite failed first.

## Why inference did not start

1. SSH succeeded with the existing project key. SSH sessions did not inherit `RUNPOD_POD_ID`; the first watchdog attempt therefore rejected the absent identity.
2. The operator read the container's own PID-1 environment in memory and passed its matching pod identity and existing pod-scoped API key to the child. No token was printed or published.
3. The installed legacy CLI's read check failed with `Unauthorized`; configuring that existing key also failed.
4. The official current CLI, `runpodctl 2.14.0-dd55bcf`, was installed from RunPod's GitHub release. Its single-pod read returned HTTP 403 `forbidden` using the same existing key.
5. A documented GraphQL single-pod query using the documented authentication format also returned HTTP 403. This corroborates the access problem beyond an SSH environment or legacy-CLI issue. It does not establish the exact provider-side cause or whether a stop mutation would be allowed.
6. No watchdog receipt was fabricated and no execution guard was bypassed. The operator copied the two existing failure logs, requested console shutdown, and verified stopped state independently. The modern watchdog was never spawned, so it had no log to copy.

The provider's built-in key could not satisfy this runner's required read preflight. **Stop permission itself was not tested**; a read denial must not be presented as proof of stop denial. A different credential with verified necessary permissions or a separately reviewed lifecycle implementation is required before another paid start. Do not broaden account permissions merely to make an unchanged check pass.

## Preservation and interpretation

Executed setup source: `9c311be8312e37aa5e2ed5545fb5c9ee708468e8`. Reviewed runtime-plan SHA-256: `6bcbee08e0bdb30f148f6a7a84c75afdcd327dc45c94b03b82724f0b625f1ee1`. Original inputs, 40-development/18-validation membership, ABBA ordering, prompts, model revision, framing/schema and decoding are unchanged.

Zero model downloads, loads or generations occurred. No MRI data, reports, IDs, raw responses, credentials, model weights or host connection details are included here. Private approval records and setup logs remain local. The temporary remote checkout was disposable and the exact source/preparation already existed locally.

The next step is to validate the provider authorization path **before renting again**. This outcome cannot assess the evidence-first prompt, multilingual extraction, agreement or medical accuracy.

Provider references: [CLI setup and built-in pod keys](https://docs.runpod.io/runpodctl/overview), [GraphQL pod operations](https://docs.runpod.io/sdks/graphql/manage-pods), [official CLI releases](https://github.com/runpod/runpodctl/releases).

## Local recovery preparation

A separate [stopped-pod preflight](../../scripts/runpod_stopped_preflight.py) queries only the supplied Pod ID, requires its state to be EXITED, then requests stop for that same already-stopped Pod and verifies the returned identity/state. It cannot create, start or delete resources. It reads a credential from `RSNA_RUNPOD_CONTROL_TOKEN`, suppresses authenticated URLs/provider error details and prints a secret-free receipt. A successful stopped-pod operation would establish acceptance of that request, not guarantee future shutdown of a running pod. Provider rejection remains a failure; the helper does not retry or alter state to force a pass.

Six [unit tests](../../tests/test_runpod_stopped_preflight.py) cover exact targeting, order, rejection of running/mismatched targets, missing credentials, mismatched stop responses and suppression of secrets in error messages. These use fake transports, not provider requests.

The console draft is named `rsna-v3-temporary-control`: Restricted, GraphQL Read/Write, AI API None. RunPod warns that GraphQL write access can manage account resources; the UI exposes no per-Pod write restriction here. No key has been created. Explicit user approval of this access expansion is pending. If approved, validate it locally with the GPU stopped, keep it out of project artifacts/logs, use it only for the approved experiment's lifecycle and revoke it afterward. No new model recipe or relaxed execution gate is proposed.
