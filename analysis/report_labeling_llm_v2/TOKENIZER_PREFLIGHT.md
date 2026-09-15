# CPU tokenizer preflight — 2026-09-15

**Tokenization measured; inference NOT RUN.** The same five original development reports were processed locally with real Transformers tokenizers. No model backend, generation, weights, GPU startup, paid compute, training or validation inference was used. The frozen v1 folders, original 40/18 split, prompts, parser and semantic acceptance gate remain unchanged.

| Check | MedGemma saved export | Qwen pinned snapshot |
|---|---:|---:|
| Reports tokenized | 5 | 5 |
| Minimum input tokens | 1,618 | 1,670 |
| Maximum input tokens | 1,945 | 2,024 |
| Total input tokens | 8,504 | 8,798 |
| Maximum input plus 2,048 output reserve | 3,993 | 4,072 |
| 8,192 input cap | Pass | Pass |
| Rendered text / actual token IDs | Equal on all five | Equal on all five |
| Pinned template and EOS/PAD | Match | Match |
| Asset revision provenance | **Unverified** | Verified pinned Hub snapshot |
| Config context budget | Unknown | Pass: 40,960 context |
| Loaded-weight/CUDA verification | Not performed | Not performed |

These counts describe instructions, definitions, the original report and chat formatting together. They are not report-only lengths. The inputs remain one Spanish and four English reports; no translation or validation-label tuning occurred. The output of this check is token IDs and audit metadata, **not twelve diagnostic answers**. Counts do not estimate accuracy, generation length, speed or cost.

## Assets and provenance

Expected models remain `google/medgemma-1.5-4b-it` at `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b` and `Qwen/Qwen3-14B` at `40c069824f4251a91eefaf281ebe4c544efd3e18`.

The default local Hub cache was absent. The preserved MRI pilot contains a complete MedGemma tokenizer and processor export, including vocabulary, SentencePiece model and template. Its adapter config identifies the base model but has `revision: null`; the tokenizer export also lacks a commit hash. The template matches the pinned v1 template exactly. Its file hashes and BOS/EOS/PAD/UNK metadata are recorded, but matching a template does **not** establish vocabulary/revision identity. We loaded the export directly; we did not relabel it as a pinned Hub snapshot. This remains a diagnostic check, not a completed pinned MedGemma preflight.

An attempt to retrieve the pinned MedGemma assets was denied because the local CLI has no authenticated access to the gated repository. Earlier browser approval does not establish local CLI authentication. No credential was printed, requested in chat, or committed. The unresolved action is to authenticate the local CLI with the already-approved account, or transfer the pinned tokenizer assets and provenance from the preserved remote cache. Starting a GPU is unnecessary for this task.

For Qwen, only four explicitly named files were downloaded: `tokenizer.json`, `tokenizer_config.json`, `config.json`, and `generation_config.json` (approximately 11.4 MB). `hf cache verify` checked all four against remote checksums at the exact revision. It reported 14 other remote files missing locally, as expected for a tokenizer-only download. This is **not a complete model cache**. The local script also verifies each content-addressed blob (Git blob SHA-1 or SHA-256). File hashes, sizes, special-token IDs and package versions are in [the aggregate receipt](aggregate/tokenizer_preflight.json). Weight blobs on the stopped remote storage were not inspected in this stage; current remote cache completeness remains unknown.

## Reproduction

Use an isolated CPU environment. Install the exact ten versions in [runtime.json](configs/runtime.json), including Transformers 5.12.0 and PyTorch 2.8.0; do not update the preserved MRI environment. The local preflight matched all ten pins. This does not verify CUDA or GPU kernels.

Download tokenizer assets separately, before the offline command. Explicit filenames avoid downloading weights:

```bash
hf download Qwen/Qwen3-14B \
  tokenizer.json tokenizer_config.json config.json generation_config.json \
  --revision 40c069824f4251a91eefaf281ebe4c544efd3e18 \
  --cache-dir state/cache/tokenizer-preflight
hf cache verify Qwen/Qwen3-14B \
  --revision 40c069824f4251a91eefaf281ebe4c544efd3e18 \
  --cache-dir state/cache/tokenizer-preflight
```

Prepare a fresh package using [the existing preparation command](README.md). The new script changes code fingerprints; older packages are preserved but stale. Then, using the isolated environment's Python:

```bash
python analysis/report_labeling_llm_v2/scripts/v2_tokenizer_preflight.py \
  --prepared state/runs/report-labeling-llm-v2-tokenizer-NEW \
  --model qwen --cache state/cache/tokenizer-preflight \
  --output state/runs/report-labeling-llm-v2-tokenizer-NEW/qwen-preflight

python analysis/report_labeling_llm_v2/scripts/v2_tokenizer_preflight.py \
  --prepared state/runs/report-labeling-llm-v2-tokenizer-NEW \
  --model medgemma --local-export state/runs/20260913T135620Z/teacher_adapter \
  --output state/runs/report-labeling-llm-v2-tokenizer-NEW/medgemma-preflight
```

The tokenizer command forces offline mode, requires the exact five input fingerprints and current preparation, checks pinned packages/template/EOS/PAD, disables truncation, compares actual tensor IDs with retokenized rendered text, and enforces input/output context budgets where config provenance exists. It never instantiates `HFBackend`, calls `generate`, unlocks execution flags, or contacts a provider. Local-export mode explicitly reports `verified_snapshot_revision: null` and unknown context. A successful receipt always records `inference_ready: false`.

Private outputs are `inputs_tokenized.json` (full rendered prompts and input IDs) and `receipt.json` (source/code hashes, metadata and aggregate counts). They remain under ignored state, with exclusive file creation and owner-only permissions; no token IDs or reports are published. Assets and earlier results are read-only. The receipt is separate from the parent-session inference contract and cannot substitute for model results in the viewer.

Transformers emitted a processor-kwargs deprecation warning for MedGemma. Actual processor IDs matched untruncated retokenization for every input, so this check found no truncation discrepancy. The warning is retained as an integration observation; runtime argument changes would require separate review and retesting.

## Remaining work

Verify the pinned MedGemma vocabulary and processor; then inspect complete weight caches and the prospective CUDA/BF16 environment before any paid smoke decision. All execution flags remain false. No new labels or accuracy results exist, and the twelve unresolved human-review items remain unresolved.

## Verification

The full repository suite passed **216 tests**, including eight new vocabulary/blob-integrity and context-budget checks. Both real tokenizer checks were repeated after resolving a new-script import-name collision; rendered prompts and token IDs were identical. The 70 protected v1 files and all 410 files present in private state at the start of this stage remained byte-identical. Local Markdown links and staged publication contents were checked; the local doctor confirmed CUDA unavailable.
