# Full cache and GPU load preflight

**Status: GPU load preflight completed; inference NOT RUN.** Both models passed after explicit authorization of the existing A100 for up to 30 minutes and $0.80 compute. Results are below.

## Reviewed plan (preserved context)

The plan was prepared locally before starting the pod. Both tokenizer checks are complete. This separate stage checks exact weight/config bytes and whether each model loads on CUDA in BF16. It produces no diagnostic predictions, loss or accuracy. All generation flags remain false.

## Scope

Input: the exact same five original development reports and pinned model revisions. Load MedGemma and Qwen sequentially in separate bounded processes; no adapters, quantization, images, translation or validation inputs. The process checks the complete required cache against [Hub file digests](configs/weight_manifest.json), including the weight index/shards, before loading anything. The manifest was retrieved through the authenticated Hugging Face API at the exact revisions; no model weights were downloaded for local preparation.

The MedGemma manifest contains two weight shards (about 8.6 GB); Qwen contains eight (about 29.5 GB). The local tokenizer-only cache correctly fails full-cache readiness for both. The preserved remote volume must be inspected after access is available; its actual current contents have not been verified. Missing or corrupt assets cause failure, with no automatic downloads or retries.

Output: private cache-audit receipts, loaded model/config revision, package versions, GPU identity, CUDA version, native BF16 support, parameter dtype/device checks, configured SDPA attention, effective stopping tokens, loaded-model context limit, five input lengths, elapsed time, and peak allocated/reserved GPU memory. Each prompt is transferred independently to CUDA and released. No model forward pass or `generate()` call occurs. Consequently, this stage does **not** verify SDPA kernel execution, KV-cache memory, generation latency or ability to complete a full answer. The model's forward/generate methods are also replaced with refusing functions after load.

The existing [HFBackend](scripts/v2_runtime.py) is reused only for its constructor and metadata; the generation engine is not called. The new [load-only command](scripts/load_preflight.py) has explicit `--execute-load` and reviewed-plan checks. Direct worker CLI invocation requires a nonce through an inherited pipe and the live parent PID. The parent validates each child result against the plan/model/code hash. These controls prevent accidental bypass, not arbitrary user-authored Python execution. This stage's receipts are separate from the inference session contract and cannot be shown as model predictions in the case viewer.

## Commands

Prepare a fresh private package using [the existing preparation command](README.md), then create a load plan:

```bash
python analysis/report_labeling_llm_v2/scripts/load_preflight.py plan \
  --prepared state/runs/report-labeling-llm-v2-load-NEW \
  --output state/runs/report-labeling-llm-v2-load-NEW/load_plan.json

python analysis/report_labeling_llm_v2/scripts/load_preflight.py audit \
  --model medgemma --cache /private/hub-cache \
  --output state/runs/report-labeling-llm-v2-load-NEW/medgemma_cache.json
```

Repeat the audit for Qwen. The audit runs on CPU, reads weights in streaming chunks, and never modifies them. It verifies hashes rather than trusting filenames, sizes or a directory named after a revision.

After resource authorization and remote setup, the load-only entry point is:

```bash
python analysis/report_labeling_llm_v2/scripts/load_preflight.py run \
  --prepared state/runs/report-labeling-llm-v2-load-NEW \
  --plan state/runs/report-labeling-llm-v2-load-NEW/load_plan.json \
  --plan-sha256 REVIEWED_LOAD_PLAN_SHA256 \
  --cache /private/hub-cache \
  --output state/runs/report-labeling-llm-v2-load-NEW/gpu_load \
  --execute-load
```

Use the exact ten runtime package pins in [runtime.json](configs/runtime.json), with CUDA-compatible PyTorch on the GPU host. Preserve the old MRI environment and all historical files. Copy the current source and verified private preparation to a new remote work directory. Never transfer credentials through Git or bake provider connection settings into these commands.

## Bounds and current resource proposal

On 2026-09-15, the RunPod console showed the preserved A100 PCIe pod stopped, with a **$1.59/hour** start quote. No pod was started during local preparation; the subsequent authorized execution is recorded below. Provider identifiers and connection details remain private.

Proposed paid window: **at most 30 minutes from pod start**, approximately **$0.80 compute**, excluding existing storage. This includes connection/setup, cache checks and both model loads. If the original cache/environment is unavailable, stop and report the missing items; do not start downloading tens of GB or migrate/deploy another pod implicitly.

The command imposes a 15-minute limit per model and a 30-minute total process limit, including cache verification. The operator must shorten the allowed remaining execution window if setup consumed part of the paid window, or stop the run/pod at the earlier billing deadline. Workers are killed/reaped on timeout/interruption. **Process termination does not stop RunPod billing.** The operator must stop the pod through RunPod immediately after completion, failure, or the 30-minute provider deadline, and verify the stopped state. A lost operator connection is not an automatic provider shutdown guarantee.

Starting paid compute requires acceptance of this concrete cost/window. Local preparation and a plan digest alone do not authorize charges. The 20-generation smoke remains outside this stage.

## Local checks

Synthetic CPU tests cover missing/corrupt shards, an incorrect weight index, Git-blob checksums, standalone-worker refusal, pipe/nonce handling, explicit execution guarding, absence of forward/generation calls, and parent success/fail-fast handling. Existing watchdog tests exercise actual process-group termination. These tests do not establish real CUDA compatibility or VRAM use. Frozen v1, the 40/18 split, prompts, ontology and acceptance gates remain unchanged.

The full local suite passed **226 tests**. The local audit verified 10 of 13 required MedGemma files and 4 of 13 required Qwen files; missing items are weight shards and their indices. All 70 frozen v1 files and 483 pre-existing private files remained unchanged. Those were preparation checks; real GPU loading was subsequently completed as recorded below.

## Measured execution — 2026-09-15

Executed source commit: `91bdbbd9e91c559449a356b895b0b29f40953c22`. Both child results and the completed parent receipt match the reviewed plan and code/input fingerprints. All 13 required cached files per model passed streamed checksums, including all weight shards and the exact index coverage. No files were automatically downloaded or repaired on the GPU host.

| Measurement | MedGemma | Qwen |
|---|---:|---:|
| Cache + tokenize + load + transfer check | 45.13 s | 91.54 s |
| Peak allocated GPU memory | 8.010 GiB | 27.508 GiB |
| Peak reserved GPU memory | 8.012 GiB | 27.512 GiB |
| Device free memory after checks | 70.752 GiB | 51.324 GiB |
| Loaded context limit | 131,072 | 40,960 |
| Largest input + 2,048 reserve | 3,993 | 4,072 |
| All parameters CUDA/BF16 | Pass | Pass |
| Native BF16 / configured SDPA | Pass | Pass |
| Model forward calls / generations | 0 / 0 | 0 / 0 |

The parent session took **140.61 seconds** including subprocess overhead. Per-model elapsed times include cache hashing and processor setup; they are not pure weight-loading or inference latency. Hardware was NVIDIA A100 80GB PCIe, CUDA 12.8, driver 595.91.07. All ten package pins matched, with the CUDA build `torch 2.8.0+cu128`. Each model ran in its own process; memory numbers are not summed. PyTorch allocation is not total device usage.

The saved environment and model cache were reused. Startup regenerated the container SSH host key; its fingerprint was independently matched against the authenticated RunPod container log before accepting the new endpoint. Strict host-key checking stayed enabled. The transfer reported unsupported ownership restoration on the network volume; all source, preparation and plan byte hashes were then verified successfully before execution. No old experiment files were overwritten.

An outer deadline alarm began from the approved provider start window and reserved one minute for stopping the pod. It would interrupt the parent so its existing supervisor could kill/reap the active worker. The checks completed well before that deadline. Results were copied locally and verified against the parent-recorded child hashes before shutdown.

RunPod was then stopped and visibly showed **Not running / $0.00 per hour**. A conservative start-request to stop-verification observation interval was about **8 minutes 11 seconds**, corresponding to approximately **$0.22 compute** at $1.59/hour, below the $0.80 cap. This is an estimate, not an invoice, and excludes storage. The original worker-session receipt still says `provider_stopped: false` because it was written before shutdown; a separate private provider receipt records the later verified stop rather than rewriting history.

[Public aggregate results](aggregate/load_preflight_results.json) contain runtime/config details and measurements. Full receipts, cache audits, logs, preparation, provider timing and the deadline wrapper are preserved privately. No weights, credentials, host addresses, study IDs or report text are published.

**Recommendation:** cache provenance and CUDA/BF16 loading are now verified for both models. Review these results before deciding on a separately bounded five-report, two-repeat-per-model generation smoke. This stage does not establish generation success, SDPA kernel execution, KV-cache requirements, repeatability, diagnostic correctness or accuracy. No training, validation inference, 20-generation smoke or bulk labeling occurred, and all generation flags remain false.
