# Protocol: Qwen live execution runtime v1

## Fixed design and source reuse

The stage is bound to prepared input plan SHA-256 `c4586bb0c65163244dac80344ea96f691493a1b89a5911eefcba54f2384c74fc` and resource proposal SHA-256 `61d0dfe522f4ad4e4d00fef86532bc55f50c989a2495c6e9d6637be036165b12`. It loads the exact five-report package at `state/runs/qwen-runtime-v1-20260915-prepared-final` using the frozen `runtime_qwen.checked_plan`; that verifier confirms the original 40-development/18-validation split and exact prompt/token artifacts. Neither the prepared reports, the 40/18 split, either prompt, either parser, `analysis/qwen_runtime_v1/`, nor `analysis/qwen_execution_v1/` is modified here.

The model recipe stays Qwen/Qwen3-14B revision `40c069824f4251a91eefaf281ebe4c544efd3e18`, offline cache only, no remote code, batch one, greedy decoding, native BF16, SDPA, input cap 8,192, output cap 2,048, 40,960 context limit, and stop IDs `[151645,151643]`. The runner calls frozen v2 `HFEncoder`, `HFBackend`, `check_versions`, cache audit, and the frozen Qwen v2/v1 parsers. It applies the frozen Qwen model configuration locally to the encoder and does not mutate frozen module globals.

## User authorization record

The production CLI requires an explicit `--execute` and the config must also be deliberately enabled in a separately reviewed change. The approval record must be a user-created JSON file with mode `0600`; the runtime never generates or edits it. Its keys must match this exact schema (no additional fields):

```json
{
  "approved": true,
  "approval_reference": "user approval reference",
  "plan_sha256": "<exact execution_plan.json SHA-256>",
  "prepared_plan_sha256": "c4586bb0c65163244dac80344ea96f691493a1b89a5911eefcba54f2384c74fc",
  "proposal_sha256": "61d0dfe522f4ad4e4d00fef86532bc55f50c989a2495c6e9d6637be036165b12",
  "approved_at": "timezone-aware timestamp",
  "provider_start_requested_at": "timezone-aware timestamp",
  "provider_deadline": "timezone-aware timestamp",
  "watchdog_stop_at": "provider_deadline minus exactly 300 seconds",
  "maximum_usd": 1.5,
  "actual_compute_usd_per_hour": 0.84,
  "actual_storage_usd_per_hour": 0.012,
  "pod_id": "exact pod id",
  "gpu_name": "NVIDIA RTX 6000 Ada Generation",
  "gpu_count": 1,
  "cloud_type": "SECURE",
  "container_disk_gb": 80,
  "persistent_volume_gb": 0,
  "network_volume_id": null
}
```

The validator also enforces a non-expired proposal quote, `approved_at <= provider_start_requested_at <= now < watchdog_stop_at < provider_deadline`, a provider window no longer than 3,600 seconds, compute no more than $0.84/hour, storage no more than $0.012/hour, and total estimated cost within the $1.50 approval.

## Pre-model gate order

Before cache audit, tokenizer load, or model load, the runner checks: committed execution-plan bytes/hash; prepared-plan bytes/hash and five-report identity; private authorization and exact plan/proposal hashes; exact pod/GPU/cloud/storage/rates/budget/deadlines; private same-pod stopped-state read/stop/read-back preflight; and a live watchdog whose process, command, environment, receipt, script hash, deadline, and stop-preflight hashes verify. It fails closed on any mismatch.

Before generation it checks the exact single GPU name, CUDA 12.8, torch 2.8.0+cu128 and all frozen dependency versions, native BF16, at least 36 GiB free GPU memory, the complete pinned offline cache, exact input token parity with the prepared artifacts, input/context caps, SDPA, loaded revision, and Qwen stop IDs. It does not download or fall back to another model, cache, host, or GPU.

## Per-generation records and stopping

For every planned generation, a durable `attempts.jsonl` record is flushed and fsynced before calling `generate`. The raw generation and saved output token IDs are then flushed and fsynced before parsing. Independent decode preserves all special tokens and uses `clean_up_tokenization_spaces=False`; it must match the saved full decoded string byte-for-byte. Exactly one terminal verified Qwen stop token may be removed for `raw_output`, which must also match byte-for-byte. Any incomplete generation, mismatch, missing token, parser/runtime exception, deadline boundary, or other failure ends the entire experiment immediately. No retry, repair, continuation, or favorable repeat is permitted.

The output stores the frozen primary v2 parse and secondary v1 parse. The output directory is new, private, and under ignored `state/` or an external private path. Session and per-generation files are durably written with restrictive permissions; result receipts include hashes. No organizer labels are read.

## Session and shutdown rules

The four runs execute in fixed ABBA order with five reports each, at most 20 generations. Total inference is capped at 1,800 seconds and must finish before `watchdog_stop_at`, which is exactly five minutes before the hard provider deadline. Before starting a generation, the runner also requires more than 330 seconds to the watchdog stop point, covering the frozen 300-second generation cap plus a 30-second persistence/parse margin. The operator must stop the pod immediately after success or failure rather than wait for the watchdog. Preserve, hash, and copy result artifacts first when time permits. At T−5 minutes the watchdog issues bounded stop attempts as a backstop. After stopping, independently read the exact pod state and billing rate; confirm stopped/exited and $0/hour, then revoke any temporary provider-control credential. Local process exit alone is not evidence that provider billing stopped.

## Future operator checklist

1. Confirm this exact source and plan SHA, five-report prepared plan SHA, and proposal SHA; run CPU tests and check the quote is still valid.
2. In the provider console, verify Secure Cloud, exactly one RTX 6000 Ada Generation, 80 GB container disk, no persistent or network volume, and the actual compute/storage rates. Confirm the exact pod ID. Do not use an alternate allocation.
3. Create the exact approval record above yourself with `0600` permissions and the exact execution-plan SHA. Keep it outside Git and outside the output package.
4. Before inference, while the pod is already stopped, execute the reviewed same-pod read/stop/read-back preflight using the same CLI and credential context. Save its private receipt. This preflight must not start the pod.
5. Only after separately deciding to incur the bounded cost, start the already approved exact pod; set `RUNPOD_POD_ID` to its exact ID. Launch the reviewed watchdog with the private receipt paths and fixed deadline/stop time, then confirm the live receipt/process.
6. On the pod, verify the source hashes, grant, pod identity and live watchdog before accessing model cache. Invoke the runner with both plan hashes and `--execute`; it still requires the source policy to have been separately enabled. Do not extend the deadline or scope.
7. Stop immediately on completion or any failure. Copy/hash artifacts where possible, independently verify stopped/exited state and $0/hour, then revoke temporary control credentials. Preserve partial receipts as failed/incomplete; never resume them.

This document describes a future operator flow. This implementation milestone does not create approval, start or stop a provider resource, load a model, or authorize spending.
