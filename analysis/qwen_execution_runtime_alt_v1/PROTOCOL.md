# Protocol: alternate-GPU Qwen live runtime v1

## Fixed scientific scope

This runner is bound to the frozen five-report prepared package and exact prepared-plan SHA-256 `c4586bb0c65163244dac80344ea96f691493a1b89a5911eefcba54f2384c74fc`. It also pins prior execution plan `0223e5be543e0bb8f4493302476ade6cf53b5ee08653acf42858bd8263e249c5` and alternate-resource proposal `884dc38d2004abb44c64d2363e525969bf7f7fabd35f86eb5d03144bf34e72e2`.

The experiment remains Qwen/Qwen3-14B revision `40c069824f4251a91eefaf281ebe4c544efd3e18`, native BF16, SDPA, batch size one, the same five development reports, control-1 → candidate-1 → candidate-2 → control-2, five reports per block, and at most 20 generations. There is no training, validation inference, full-development inference, bulk extraction, retry, or repair. Prompts, parsers, report package, and 40/18 split remain in their frozen directories.

The versioned GPU allowlist is exactly: NVIDIA A40, NVIDIA RTX A6000, NVIDIA L40, NVIDIA RTX 6000 Ada Generation, and NVIDIA L40S. One name from this list must be selected in the private grant. The runtime checks the grant's selected name against the allowlist and checks the visible single GPU against that exact grant. Availability of another listed GPU is never used as fallback.

## Private approval and provider observation

Execution requires the config switch and `--execute`; the checked-in switch remains false. Before loading tokenizer/cache/model, the runner requires all of the following private local records, each a regular file with mode `0600`: user approval, exact-pod stopped-state stop-preflight, armed watchdog receipt, and signed-in provider observation. The user creates the approval record; this runtime never creates it.

The approval JSON has exactly the following fields:

```json
{
  "approved": true,
  "approval_reference": "user approval reference",
  "execution_plan_sha256": "<exact configs/execution_plan.json SHA-256>",
  "prepared_plan_sha256": "c4586bb0c65163244dac80344ea96f691493a1b89a5911eefcba54f2384c74fc",
  "resource_proposal_sha256": "884dc38d2004abb44c64d2363e525969bf7f7fabd35f86eb5d03144bf34e72e2",
  "pod_id": "exact existing pod ID",
  "gpu_name": "one exact reviewed allowlist name",
  "gpu_count": 1,
  "cloud_type": "SECURE",
  "region": "one exact signed-in region",
  "container_disk_gb": 80,
  "persistent_volume_gb": 0,
  "network_volume_id": null,
  "actual_compute_usd_per_hour": 0.49,
  "actual_storage_usd_per_hour": 0.012,
  "maximum_usd": 1.5,
  "approved_at": "timezone-aware timestamp",
  "provider_start_requested_at": "timezone-aware timestamp",
  "provider_deadline": "timezone-aware timestamp",
  "watchdog_stop_at": "provider deadline minus exactly 300 seconds"
}
```

The compute rate must not exceed the selected GPU's ceiling: A40 $0.49/h, RTX A6000 $0.53/h, L40 $0.82/h, RTX 6000 Ada $0.84/h, or L40S $1.09/h. Storage must be at most $0.012/h. The total maximum is $1.50, and the provider window is at most 3,600 seconds. The proposal freshness deadline is enforced.

The separate provider observation JSON has exactly the resource fields plus `observed_at` and `provider_state`. It must equal the grant resource fields field-for-field and record the exact pod as `STOPPED`. Capture it from the signed-in provider state without changing the pod, after the grant is approved and within 10 minutes before the recorded provider start request:

```json
{
  "pod_id": "exact existing pod ID",
  "gpu_name": "one exact reviewed allowlist name",
  "gpu_count": 1,
  "cloud_type": "SECURE",
  "region": "one exact signed-in region",
  "container_disk_gb": 80,
  "persistent_volume_gb": 0,
  "network_volume_id": null,
  "actual_compute_usd_per_hour": 0.49,
  "actual_storage_usd_per_hour": 0.012,
  "observed_at": "2026-09-24T14:04:00+00:00",
  "provider_state": "STOPPED"
}
```

Rates and timestamps in these examples are illustrative schema values only; a future record must use the actual signed-in values. The runtime enforces `approved_at <= observed_at <= provider_start_requested_at <= now`, requires the observed state to be exactly `STOPPED`, and caps the observation-to-start interval at 600 seconds. Once start is requested, the observation is historical pre-start evidence only; it is not reused to claim that the pod is still stopped. The same-pod preflight and live watchdog establish the later control path. The authorization validator reuses the reviewed proposal gate for allowlist, per-GPU ceiling, storage, budget, quote, and deadline checks. It first strictly checks that `execution_plan_sha256` is the exact new plan SHA, then adapts only an in-memory copy of that field for the proposal gate, whose frozen proposal is keyed to the previous plan. The grant on disk is never rewritten.

The watchdog is the reviewed exact-pod watchdog. It remains bound to the approved pod and timestamps; the live runtime revalidates its PID, command, environment, receipt, and preflight before every generation. The observation receipt is not a capacity guarantee; it only binds the provider facts captured for the exact selected pod. If that pod cannot start, stop without trying another resource.

## Runtime preflight and generation safeguards

Before model loading, checks include: exact new-plan SHA and bound source/policy/gate/watchdog hashes; old execution-plan SHA; alternate proposal SHA; prepared package identity; exact private approval and matching provider observation; same-pod stopped-state preflight; and live verified watchdog. No provider API is called by this runner.

Before inference, the worker checks exactly one visible GPU and exact name equality with the grant; at least 36 GiB free VRAM; native BF16; CUDA 12.8; torch 2.8.0+cu128 and all pinned packages; and records NVIDIA driver version. It audits the complete pinned offline Qwen cache, exact input-token parity with prepared artifacts, 8,192 input cap, 40,960 context limit, SDPA, model revision, and stop IDs `[151645,151643]`. It does not download, quantize, offload, change batch size, or change the model recipe.

The full session runs as a parent-attested child process group under the frozen hard supervisor. Its timeout is below 1,800 seconds and bounded by remaining time before the watchdog stop, with a launch margin. The worker verifies watchdog liveness immediately before each generation. It writes/fsyncs attempt receipt, then raw output/token IDs, then frozen v2 primary and v1 secondary parses. Saved output IDs are independently decoded with special tokens preserved and `clean_up_tokenization_spaces=False`; both full decoded text and the raw text after removing exactly one terminal verified Qwen EOS must match byte-for-byte. An incomplete generation or any mismatch ends the run without retry.

No output directory may be overwritten or resumed. Failures and timeouts retain hashes of partial artifacts. The operator should stop the exact pod immediately on success or failure, copy/hash outputs when possible, independently verify stopped state and $0/hour, and revoke any temporary control credential. The watchdog fires at T−5 minutes as the stop backstop.

## CLI and enablement

The future command requires `--execute`, an explicitly reviewed enablement change, exact hashes, private approval/watchdog/preflight/provider-observation records, the prepared package, offline cache, and a new output directory. Example only:

```sh
poetry run python analysis/qwen_execution_runtime_alt_v1/scripts/qwen_live.py \
  --execute \
  --prepared state/runs/qwen-runtime-v1-20260915-prepared-final \
  --prepared-plan-sha256 c4586bb0c65163244dac80344ea96f691493a1b89a5911eefcba54f2384c74fc \
  --execution-plan analysis/qwen_execution_runtime_alt_v1/configs/execution_plan.json \
  --execution-plan-sha256 <exact-plan-sha256> \
  --authorization <private-0600-approval.json> \
  --watchdog-receipt <private-0600-watchdog.json> \
  --stop-preflight <private-0600-preflight.json> \
  --provider-observation <private-0600-resource-observation.json> \
  --cache <complete-offline-qwen-cache> \
  --output state/runs/<new-private-run-id>
```

This source milestone does not enable or execute the command and does not authorize spending.
