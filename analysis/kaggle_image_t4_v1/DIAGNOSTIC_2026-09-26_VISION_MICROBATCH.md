# T4 vision microbatch diagnostic — 26 September 2026, 22:57 SGT

Status: **FAILED_NONFINITE_LOGITS**, not KAGGLE_RUN_PASS. No submission CSV and no competition submission.

## Change and execution

Source commit: `2e65235cefe164b62009df88c6b487819942fd3a`.
Kaggle private version **7**, script version **353022704**. User requested correction and a new run after version 6, retaining the 120-minute ceiling. Request T0: 2026-09-26 14:57:35.569 UTC; hard deadline: 16:57:35.569 UTC.

Version 6's OOM was addressed by scheduling one image per call to the original SigLIP vision tower, concatenating all features in order, then using the unchanged projector/language model. This is an explicit execution change. The 6-image study, 448px slice rendering, original processor's 896px tensors, eager attention, NF4/double quantization, FP16 compute, LoRA adapter and placement policy remain unchanged. No weights were merged or overwritten.

54 focused CPU tests passed, including 18 vision scheduling tests using a tiny real SigLIP model and an Accelerate hook, plus process isolation/failure tests. These are software checks, not numerical parity for the full checkpoint.

## Actual result

| Measurement | Observed |
|---|---:|
| Kaggle total run duration | 333.6 seconds |
| Asset verification | 208.261 seconds, 32 files |
| Offline install | 17.404 seconds |
| Model load | 41.272 seconds |
| Load plus failed diagnostic | 55.706 seconds |
| Single-GPU model load | Completed |
| First complete model forward | Returned logits, failed finite-logit check |
| GPU 0 peak allocated / reserved during diagnostic | 4.774 / 5.662 GiB |
| GPU 0 free at failure | 8.667 GiB |
| GPU 1 allocated / reserved peak | 0 / 0 GiB |
| Free after child process exit, GPU 0 / 1 | 14.460 / 14.460 GiB |
| Two-GPU attempt | NOT RUN: failure was not CUDA OOM |
| Repeated-input diagnostic | Incomplete: first input failed |
| Complete example studies | 0 of 3 |
| Submission | NOT CREATED |

The OOM barrier is removed for the first forward. The current blocker is nonfinite native No/Yes logits; the record does not identify which intermediate module first produced them. It does not establish whether they were NaN, Inf, or both. Do not infer medical performance or repair the score.

Relevant traceback (study identifier omitted):

```text
submission_runtime.py:569 prepare_t4_teacher -> diagnostic
submission_runtime.py:432 diagnostic -> score_logits
submission_runtime.py:410 score_logits -> yes_probability_from_logits
inference_core.py:265
ContractError: Native No/Yes logits contain NaN or Inf
```

Actual placement: `hf_device_map = {"": "0"}`. Inputs: pixel_values `[6, 3, 896, 896]` FP16; input_ids, attention_mask and token_type_ids `[1, 1747]` INT64, all GPU 0. Quantized weights report UINT8 storage with FP16 compute; vision/unquantized parameters FP16; LoRA parameters FP32. All 272 LoRA tensors were attached. Python 3.12.13, Torch 2.10.0+cu128, CUDA 12.8, two visible Tesla T4s. Original dependency pins passed.

## Preservation and shutdown

13 output artifacts plus their ZIP were copied to private local storage and SHA-256 hashed. Executed inference_core.py, submission_runtime.py, isolated_runner.py and inference_config.json match the local source byte-for-byte. Historical runs remain untouched.

Kaggle showed **0 Active Events** and **Draft Session off** after failure. Displayed quota changed from **00:17 to 00:23 / 30 hours**: approximately six displayed minutes, rounded by Kaggle; not an exact billing measurement. No restart or third placement attempt occurred. Notebook and assets remain private, internet was off.

Notebook SHA-256: `42f4a08e49e838471a7dc1f4b4aea7c9fac2fa1e2ac3c28f6849c6009a99cb1e`.
Unchanged 32-file asset manifest SHA-256: `4b4e9396301bd0c80a6a4ee1fd169e5b800d482f8833dcd28e3cf0b92ab4e0d7`.

## Remaining blocker

**NUMERICAL_PARITY_UNVERIFIED**. Before another GPU attempt, add first-nonfinite module tracing to distinguish vision, projector, language/LoRA and output-head overflow. Only the identified sensitive operation should be considered for a documented FP32 change; no score clipping, nan_to_num, neutral fallback, arbitrary precision change, or automatic new run.
