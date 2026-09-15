# Full cache and GPU load preflight

**Status: prepared locally; GPU load NOT RUN.** Both tokenizer checks are complete. This separate stage checks exact weight/config bytes and whether each model loads on CUDA in BF16. It produces no diagnostic predictions, loss or accuracy. All generation flags remain false.

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

On 2026-09-15, the RunPod console showed the preserved A100 PCIe pod stopped, with a **$1.59/hour** start quote. No pod was started during local preparation. Provider identifiers and connection details remain private.

Proposed paid window: **at most 30 minutes from pod start**, approximately **$0.80 compute**, excluding existing storage. This includes connection/setup, cache checks and both model loads. If the original cache/environment is unavailable, stop and report the missing items; do not start downloading tens of GB or migrate/deploy another pod implicitly.

The command imposes a 15-minute limit per model and a 30-minute total process limit, including cache verification. The operator must shorten the allowed remaining execution window if setup consumed part of the paid window, or stop the run/pod at the earlier billing deadline. Workers are killed/reaped on timeout/interruption. **Process termination does not stop RunPod billing.** The operator must stop the pod through RunPod immediately after completion, failure, or the 30-minute provider deadline, and verify the stopped state. A lost operator connection is not an automatic provider shutdown guarantee.

Starting paid compute requires acceptance of this concrete cost/window. Local preparation and a plan digest alone do not authorize charges. The 20-generation smoke remains outside this stage.

## Local checks

Synthetic CPU tests cover missing/corrupt shards, an incorrect weight index, Git-blob checksums, standalone-worker refusal, pipe/nonce handling, explicit execution guarding, absence of forward/generation calls, and parent success/fail-fast handling. Existing watchdog tests exercise actual process-group termination. These tests do not establish real CUDA compatibility or VRAM use. Frozen v1, the 40/18 split, prompts, ontology and acceptance gates remain unchanged.

The full local suite passed **226 tests**. The local audit verified 10 of 13 required MedGemma files and 4 of 13 required Qwen files; missing items are weight shards and their indices. All 70 frozen v1 files and 483 pre-existing private files remained unchanged. No real GPU load has occurred.
