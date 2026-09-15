# Local inference adapter — model execution NOT RUN

The adapter is implemented and tested with synthetic backends on CPU. No v2 model/tokenizer was loaded, no weights or tokenizer vocabulary downloaded, and no GPU or inference service contacted. Reading the public Qwen configuration for template verification was a small documentation/metadata request, not tokenizer preflight. Real Transformers/CUDA integration and the current five-prompt token counts remain unverified.

The earlier local candidate is preserved at commit `68149d994e16a03802e1c4261a738f94a133661b`. This subsequent implementation changes runtime tooling, not condition definitions, prompts, acceptance rules, the 40/18 split or v1 results. Old prepared packages stay intact but become stale under the new code fingerprints.

## Input and output

Input is the exact original five development reports, in the same order, with their original languages. Each model receives one user message containing the common instructions, target definitions and a JSON-quoted report. Organizer answers and study IDs never enter the model prompt. There are no MRI images, translations, examples chosen from validation, generated repairs or training updates.

The planned session runs MedGemma repeat 1, MedGemma repeat 2, Qwen repeat 1, Qwen repeat 2: at most 20 primary generations. Each repeat loads in a separate process. A failure stops the session; there is no implicit retry or selection of a preferred repeat. A completed generation with invalid JSON remains a technical failure but may be followed by the next report; a noncompleted generation stops the run immediately.

Each future run saves:

- `start_manifest.json` and final `run_manifest.json`, with version, identity, timing, completion status, reviewed plan SHA, session ID, run-order index and hashes binding the parent start/dispatch.
- `tokenizer_preflight.json` for exactly five inputs and `runtime_metadata.json` for the loaded configuration and GPU.
- `raw_generations.jsonl`: exact rendered input, input/generated token IDs, output decoded with special tokens, response with only one terminal stopping token removed, status and timing. Written and flushed to disk before parsing.
- `predictions.jsonl`: the original record plus the unchanged v2 normalization/parser result, with separate raw text, normalized text and accepted rows.

Only generated tokens after the verified input prefix are decoded. No reasoning/prose is stripped. For parser input, remove only a single final ID in the verified model EOS set. Decode all remaining IDs with special tokens preserved and text cleanup disabled; unexpected internal control/reasoning tokens cannot disappear through broad special-token removal. The full token sequence is also decoded without removal for audit. This conservative decoding rule is prospectively recorded; v1 is not reprocessed. Synthetic backend artifacts are explicitly marked and rejected by the measured-results viewer.

## Template and stop-token review

| Item | MedGemma | Qwen |
|---|---|---|
| Input API | `AutoProcessor.apply_chat_template`, structured text content | Tokenizer chat template, string content |
| Generation prompt | Enabled | Enabled |
| Thinking control | No supported switch claimed or passed | `enable_thinking=False` passed explicitly |
| Tokenizer EOS / PAD | 1 / 0 | 151645 / 151643 |
| Model generation EOS IDs | [1, 106] | [151645, 151643] |
| Effective PAD | 0 | 151643 |

Exact model revisions remain in the existing [configs](configs/medgemma.json) and [Qwen config](configs/qwen.json). The [runtime policy](configs/runtime.json) pins template hashes, stop-token expectations and package versions. Qwen's template hash was independently checked against its [official pinned tokenizer configuration](https://huggingface.co/Qwen/Qwen3-14B/blob/40c069824f4251a91eefaf281ebe4c544efd3e18/tokenizer_config.json). MedGemma's template/hash and token metadata come from the preserved pinned v1 runtime; a fresh offline processor check is still required. Its [official model card](https://huggingface.co/google/medgemma-1.5-4b-it) supports the processor-template construction and generated-token slicing used here.

A real load must match the pinned template hash, tokenizer tokens, model revision and model stopping tokens. The exact rendered text is retokenized without added special tokens and compared with processor input IDs, preventing an unreported BOS/template mismatch. Both paths disable input truncation. All five inputs are tokenized before weights load; any input over 8,192 tokens stops preflight. After model loading, input plus the 2,048 output-token budget must fit the verified model context limit.

One deliberate runtime correction: v1 used `model_pad or tokenizer_eos`, replacing valid MedGemma PAD 0 with EOS 1. V2 checks for `None`, preserving PAD 0. This was tested synthetically; its numerical effect on generation is not measured. Do not silently describe the two runtime recipes as identical.

Qwen's [official card](https://huggingface.co/Qwen/Qwen3-14B) documents its non-thinking switch. Greedy decoding is retained from the prospective paired comparison; this is an experimental control, not a claim to follow the card's suggested sampling defaults. No equivalent MedGemma switch is claimed.

## Limits and failure handling

The effective recipe is BF16, batch one, greedy decoding, one beam, one returned sequence, SDPA and seed 20260914. Stop IDs are model-specific; valid PAD 0 is preserved. EOS must terminate a nonempty generated sequence within the token/time limits. An EOS followed by more generated tokens, a missing EOS, or a reached output cap without EOS is not completed. Timeout takes precedence even if EOS arrived late. OOM, context overflow, runtime failure and truncation never become negative labels or repaired generations.

[Transformers generation documentation](https://huggingface.co/docs/transformers/v5.12.0/en/main_classes/text_generation) describes `max_time` as finishing the current pass after the allotted time. It is not a hard watchdog. The parent therefore bounds each subprocess, including tokenizer/model loading, by 1,800 seconds and the whole session by 3,600 seconds. On timeout or parent interruption, the owned worker process group is killed immediately and the direct child reaped; tests also check descendant termination and partial artifacts remain visibly incomplete. These limits stop local processes, **not provider billing**; an operator/provider stop remains necessary. No provider provisioning API is included.

The adapter imports Transformers only behind the execution guard. Before importing Transformers, an explicit cache must contain the requested pinned snapshot. The child sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, all relevant cache roots to that supplied path and implicit token use off. Missing snapshots fail before loading; there is no default-cache fallback. Model and processor loads always use `local_files_only=True`, pinned revisions and `trust_remote_code=False`. It has no automatic download option. Ten critical package versions, including tokenizer/template libraries, are pinned from the preserved v1 environment and checked against the runtime policy before loading, avoiding an implicit upgrade of the preserved MRI environment. Existing cached files must be available and independently transferred/downloaded under a future authorized setup step.

## Reviewable commands

Local preparation and plan creation are safe CPU operations. Restore the private state, choose a fresh path, and run from the repository root:

```bash
python analysis/report_labeling_llm_v2/scripts/prepare_local.py \
  --output state/runs/report-labeling-llm-v2-adapter-NEW
python analysis/report_labeling_llm_v2/scripts/run_smoke.py plan \
  --prepared state/runs/report-labeling-llm-v2-adapter-NEW \
  --plan state/runs/report-labeling-llm-v2-adapter-NEW/execution_plan.json
```

The plan reports `DRY_PLAN_ONLY`, zero model calls and tokenizer preflight `NOT_RUN`. It verifies the original five-input byte hash, original source/split anchors, exact prompt contents and current candidate fingerprints. The plan is private, contains study IDs, and must not be committed. It creates no model outputs or token estimates.

Proposed future command, **locked and not executed in this stage**:

```bash
python analysis/report_labeling_llm_v2/scripts/run_smoke.py run \
  --prepared state/runs/report-labeling-llm-v2-adapter-NEW \
  --plan state/runs/report-labeling-llm-v2-adapter-NEW/execution_plan.json \
  --plan-sha256 REVIEWED_PLAN_SHA256 \
  --cache /private/model-cache \
  --execute
```

`--execute` alone cannot bypass the current false execution flags in both model configs, output policy and runtime policy. A separately authorized, reviewed runtime version must enable them; that changes fingerprints and requires fresh private preparation/plan creation. Before that decision, resolve cache availability, offline processor/tokenizer checks, current GPU price and the provider stop plan. No GPU compatibility or token-budget success is claimed from fake tests.

A worker requires a one-use nonce received through an inherited anonymous pipe, matching the active parent PID and hashed dispatch; standalone `_worker` invocation is refused. The parent verifies each child before proceeding, and its completed session manifest records every child manifest hash in the prescribed order. This prevents accidental CLI bypass; it is not a security boundary against arbitrary local code. The dry plan contains no `compute_authorized` boolean: execution flags and explicit future resource authorization remain separate from a plan digest.

After an eventual completed four-run session, use `smoke_review.py --runs PREPARED/adapter_runs` to render measured outputs. The viewer requires the completed parent session and reviewed plan, then verifies all four child identities/order/hashes against it. Any partial, standalone, stale, synthetic, reordered or mixed-session run is rejected. All prior human approvals become stale if the experiment fingerprint changes. The same exact-repeat and qualified-human-review gates still apply before considering any expansion.

## Local verification

The full suite passed **208 tests**, including 41 new adapter/watchdog checks in addition to the earlier 167. Tests use fake encoders/models and CPU tensors; two short subprocess tests exercise actual termination/reaping, including a descendant. Empty-cache rejection and offline environment assignment are tested without a Transformers installation. They do not establish real-cache completeness, GPU compatibility, throughput or semantic correctness.

## Published-source review follow-up

Review of the first adapter commit (`1b4fd37`) identified that direct worker invocation could bypass the parent watchdog after a future unlock, and that the viewer did not require parent-session provenance. Both are fixed locally without enabling execution. New tests cover standalone refusal, pipe/nonce verification, parent dispatch order, fail-fast receipts and mixed/reordered session rejection. Cache checks pin template/EOS/PAD metadata and requested revisions; they do not claim full vocabulary, BOS/UNK/additional-special-token or weight-file byte verification. Cache-content provenance remains a future real-cache review item.
