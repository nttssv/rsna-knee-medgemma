# Resource alternative proposal protocol

This proposal supports the same fixed Qwen3-14B smoke identified by base execution-plan SHA-256 `0223e5be543e0bb8f4493302476ade6cf53b5ee08653acf42858bd8263e249c5` and prepared-plan SHA-256 `c4586bb0c65163244dac80344ea96f691493a1b89a5911eefcba54f2384c74fc`. It must preserve Qwen/Qwen3-14B revision `40c069824f4251a91eefaf281ebe4c544efd3e18`, the five fixed development reports, control-1 → candidate-1 → candidate-2 → control-2, five reports per pass, and at most 20 generations. No training, validation, full-40 inference, bulk extraction, retries, or repairs are authorized. BF16, SDPA, token limits, contexts, stop IDs, prompts, parsers, and tokenization remain fixed.

## Exact proposed allowlist

`configs/resource_proposal.json` is authoritative for the allowlist. Its exact `provider_name` strings are:

- `NVIDIA A40`
- `NVIDIA RTX A6000`
- `NVIDIA L40`
- `NVIDIA L40S`
- `NVIDIA RTX 6000 Ada Generation`

One and only one of these must be explicitly selected in the user's private grant. There is no automatic hardware fallback. The live runtime must require the exact approved `gpu_name`, then compare the worker's `nvidia-smi`/CUDA-reported name and visible GPU count against it. Any unlisted model, second GPU, changed precision, or architecture-specific recipe change must fail before model loading.

The original RTX 6000 Ada is retained in the allowlist to represent the current proposal. The four other devices are reviewed alternatives. All have 48 GB and BF16-capable NVIDIA architecture. The cost-preferred fallback order for a future explicit choice is A40, RTX A6000, L40, then L40S; availability and region must be rechecked at that time. This ordering is information only and must never trigger an automatic selection.

## Resource boundary

- RunPod Secure Cloud only; the private grant binds the exact selected region and pod ID.
- Exactly one listed GPU, 48 GB VRAM, with the grant binding its canonical provider name and actual signed-in compute rate.
- Container disk exactly 80 GB; persistent volume 0 GB; network volume ID null.
- Actual compute rate must not exceed the chosen GPU's per-entry Secure Cloud rate ceiling. Actual storage must not exceed $0.012/hour.
- Maximum approved total remains $1.50; provider session at most 3,600 seconds; watchdog stop exactly 300 seconds before the deadline.
- The operator stops immediately after completion or failure. No start, inference, retry, or resource modification is performed by this proposal milestone.

The highest listed candidate is L40S at $1.09/hour. Including the $0.012/hour storage allowance, a full hour is estimated at $1.102, within $1.50. The live grant binds the exact displayed compute and storage rates; values are checked against the selected allowlist row, not merely against a shared maximum.

## Required private grant and runtime gate

The future private 0600 grant must include exactly these resource bindings in addition to approval and fixed-window provenance:

```json
{
  "approved": true,
  "approval_reference": "user-supplied reference",
  "execution_plan_sha256": "future separately reviewed execution plan SHA-256",
  "prepared_plan_sha256": "c4586bb0c65163244dac80344ea96f691493a1b89a5911eefcba54f2384c74fc",
  "resource_proposal_sha256": "SHA-256 of this versioned proposal JSON",
  "pod_id": "exact provider pod ID",
  "gpu_name": "one exact provider_name from this proposal's allowlist",
  "gpu_count": 1,
  "cloud_type": "SECURE",
  "region": "exact signed-in provider region",
  "container_disk_gb": 80,
  "persistent_volume_gb": 0,
  "network_volume_id": null,
  "actual_compute_usd_per_hour": 0.0,
  "actual_storage_usd_per_hour": 0.0,
  "maximum_usd": 1.50,
  "approved_at": "timezone-aware timestamp",
  "provider_start_requested_at": "timezone-aware timestamp",
  "provider_deadline": "timezone-aware timestamp",
  "watchdog_stop_at": "exactly provider_deadline minus 300 seconds"
}
```

The numeric zeros above are schema placeholders only; a future grant must carry the actual verified nonzero provider rates. The private grant is not created in this proposal. Before model load, the future runtime must:

1. Verify the grant is a regular 0600 file, user-approved, and bound to the exact future execution plan, frozen prepared plan, and this proposal SHA.
2. Validate the one chosen GPU against the exact allowlist, count 1, Secure cloud, exact region/pod, 80 GB container disk, no persistent or network volume, exact actual rates, chosen-GPU rate ceiling, total $1.50 limit, 60-minute provider ceiling, and T−5-minute watchdog stop.
3. Compare a read-only signed-in provider observation with every grant resource field, including exact `gpu_name`, pod ID, and compute/storage rates. A generic inventory card is not a pod-specific capacity assertion.
4. Start and continuously verify the already separately approved same-pod watchdog and stopped-pod preflight, all before model/cache loading, using the reviewed runtime's current safety rules.
5. Pass the unchanged CUDA 12.8, torch 2.8.0+cu128, native BF16, ≥36 GiB free memory, SDPA, model-cache, token parity, and context checks. Enforce exact observed GPU identity and stop on any mismatch.

`scripts/resource_alt_gate.py` is an isolated CPU-verifiable implementation of these resource checks. It is not wired into or substituted for the current live runner. The current runtime remains exact-RTX-6000-Ada-only and will reject all alternate GPUs because its proposal/source hashes are frozen. A separately versioned execution runtime and execution-plan hash must integrate this gate and receive independent source review before anyone can use an alternate GPU.

## Compatibility basis and uncertainty

NVIDIA classifies A40 and RTX A6000 at compute capability 8.6 and the three Ada devices at 8.9. CUDA documents BF16 at CC 8.0 and above; CUDA's architecture matrix lists Ampere and Ada toolkit support as ongoing. This establishes architectural compatibility for the existing CUDA 12.8 / PyTorch 2.8 stack without changing precision or attention configuration. It does not prove the exact RunPod image, driver, or PyTorch SDPA kernel behavior on a particular host; those checks remain mandatory.

All cards have a nominal 48 GB. The prior runtime's measured ~27.99 GiB peak and the unchanged 36 GiB free-memory preflight make fitting plausible, not proven. We have no measured Qwen throughput on A40, RTX A6000, L40, L40S, or an alternative Ada pod. A GPU could fail the 30-minute hard inference process timeout; failure means stop and preserve partial evidence, never retry or switch hardware. Because GPU architecture can change floating-point behavior, results must record the one selected model and should not be presented as bitwise cross-GPU replication.

Public pricing references: [RunPod GPU and container storage pricing](https://www.runpod.io/pricing), [NVIDIA GPU compute capability](https://developer.nvidia.com/cuda/gpus), [CUDA BF16 requirement](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/05-appendices/mathematical-functions.html), and [CUDA toolkit/architecture support](https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html). NVIDIA's official data sheets for [A40](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a40/proviz-print-nvidia-a40-datasheet-us-nvidia-1469711-r8-web.pdf), [L40](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/support-guide/NVIDIA-L40-Datasheet-January-2023.pdf), [L40S](https://www.nvidia.com/en-us/data-center/l40s/), [RTX A6000](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/proviz-print-nvidia-rtx-a6000-datasheet-us-nvidia-1454980-r9-web%20%281%29.pdf), and [RTX 6000 Ada](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/rtx-6000/proviz-print-rtx6000-datasheet-web-2504660.pdf?ncid=no-ncid) support GPU generation and memory facts.
