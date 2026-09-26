# Completed T4 example inference — 26 September 2026, 23:19 SGT

Status: **KAGGLE_RUN_PASS**, private Kaggle version **8**, script version
**353028229**. All three visible test studies completed; 36 finite scores,
three unique test IDs and the exact 13-column submission schema passed both
Kaggle-side and local validation. **Not submitted; not scored.**

Source commit: `a0bfdb9cbefac33858af0fa47dca444053f235c3`.
Request T0: `2026-09-26T15:19:22.015Z`; approved hard deadline:
`2026-09-26T17:19:22.015Z`. The single authorized run finished in 625.5 seconds,
well inside the 120-minute maximum. Notebook and assets remained private;
Internet was OFF and the existing offline wheelhouse/model package was used.

## Correction and evidence

Version 7 stopped on nonfinite native No/Yes logits. The correction preserves
Gemma3 decoder post-normalization outputs and residual additions in FP32;
pre-normalization branch inputs are checked and cast to FP16. All original
weights, adapter, NF4/double quantization, eager attention, image selection,
six-image ordering, 448px slice rendering, processor resizing and scoring are
unchanged. No clipping, nan_to_num, neutral score or CPU/disk offload was used.
The prior one-image vision scheduling correction remains in place.

The first real diagnostic forward traced 3,257 module calls, with no observed
nonfinite output. It recorded 29 finite outputs beyond FP16's 65,504 range.
The first was decoder layer 5 at 70,858.1875; the maximum was decoder layer 29
at 281,899.8125. These residual outputs were FP32. This supports preserving
their range, but does not retrospectively prove the exact first failing
operation in version 7. The trace covers module outputs, not every functional
operation, and only the first diagnostic forward.

79 focused CPU tests passed before execution, including synthetic overflow,
fail-closed tracing, original norm parameter preservation, process isolation
and vision scheduling tests. The actual pinned Gemma3 meta architecture also
matched all 136 normalization targets without loading weights or using a GPU.

## Actual runtime and diagnostic

| Measurement | Observed |
|---|---:|
| Kaggle duration | 625.5 s |
| Asset verification, 32 files | 197.188 s |
| Offline dependency installation | 17.664 s |
| Model load | 43.337 s |
| Complete example wall time | 286.391 s |
| Full-study times, 12 forwards each | 90.137 / 97.400 / 98.816 s |
| Mean complete-study time | 95.451 s |
| Cumulative DICOM preprocessing counter | 10.335 s |
| Cumulative model-forward counter | 290.591 s |
| GPU 0 peak allocated / reserved | 4.776 / 5.662 GiB |
| GPU 1 peak allocated / reserved | 0 / 0 GiB |
| DICOM issue rows | 0 |
| Completed example studies / target scores | 3 / 36 |
| Submission score range | 0.0341004–0.3106944 |

The preprocessing/forward counters include the preceding diagnostic work;
they are not independent components that sum to the example wall time.
Use complete-study wall time for throughput estimates.

Actual placement was `hf_device_map = {"": "0"}`. The second visible T4 was
unused. Single-GPU execution succeeded; no OOM, cleanup retry or dual-GPU
attempt occurred. The approved OOM-only fallback policy was unchanged.

The repeated first-study ACL diagnostic produced native FP16 No/Yes logits
`23.046875 / 20.9375`, then FP32-softmax Yes score `0.10818894952535629`.
The second identical input had maximum logit difference **0.0** and score
difference **0.0**. All native selected logits and resulting scores passed
finite checks. This is one repeated diagnostic input, not a comprehensive
determinism or precision-parity study.

Actual dtype inventory:

- 400 packed NF4 weight tensors: UINT8 storage; all 400 linear modules compute
  in FP16. Quantized biases remain FP16.
- Vision and other unquantized base parameters: FP16; 272 attached LoRA tensors:
  FP32, active adapter `default`.
- 68 post-normalizations/residual additions: FP32 outputs; 68
  pre-normalizations: checked FP16 branch outputs. Original norm weights remain
  unchanged; this is a runtime computation/output change.
- Pixel inputs: FP16, `[6, 3, 896, 896]` after the original processor.
  Token IDs, masks and token types: INT64. All inputs and weights use GPU 0.

Observed environment: Python **3.12.13**, Torch **2.10.0+cu128**, CUDA **12.8**,
two Tesla T4s, each 14.562 GiB total and initially 14.460 GiB free. Installed
pins: transformers 4.57.6, huggingface-hub 0.36.0, tokenizers 0.22.1,
peft 0.20.0, accelerate 1.12.0, bitsandbytes 0.50.2, pydicom 3.0.2,
python-gdcm 3.0.26, sentencepiece 0.2.1 and safetensors 0.6.2.

## Preservation, hashes and shutdown

24 output files plus their ZIP were downloaded into private local storage and
SHA-256 hashed. Executed `inference_core.py`, `submission_runtime.py`,
`isolated_runner.py`, `numerical_runtime.py` and `inference_config.json` match
the T4 source byte-for-byte. The downloaded CSV was independently checked
against local example `test.csv` and `sample_submission.csv`; no report,
organizer label or Qwen output was used.

| Artifact | SHA-256 |
|---|---|
| Submitted notebook source, private | `ac3734740adc506eb6bc91407afb53efcf87fe9e25bb621c51389696f974b84c` |
| Frozen selected runtime configuration | `f3c0d16cee8353955d25ebeaf668d818fdcbeb45972085251a74c1905f3828d2` |
| Completed example submission CSV, private | `1f3d823c4b1e689b252c041945440dcc3685d0e5810cb7fbd91583739ef739cd` |
| Unchanged 32-file asset manifest | `4b4e9396301bd0c80a6a4ee1fd169e5b800d482f8833dcd28e3cf0b92ab4e0d7` |
| Unchanged adapter weights | `a0bbce8fade7c273dbbeeba39bf792f82eb166abf1bac3450d5100027496f448` |

Kaggle showed **0 Active Events** and **Draft Session off** after completion.
Its quota display moved from **00:23 to 00:33 / 30 hours**, approximately ten
displayed minutes, rounded by Kaggle. This is the observed account quota delta,
not a separately inferred two-GPU billing calculation. No second run or
competition submission was launched. Historical failures and outputs remain
unchanged.

## Practical limit

Cross-precision comparison remains **NUMERICAL_PARITY_UNVERIFIED** because no
compatible BF16 reference was available. Finite outputs and valid CSV do not
establish clinical accuracy or competition performance.

A rough planning estimate with 30% headroom is
`339 + 1.30 × 95.45 × N` seconds: about 3.54 hours for 100 studies. The hidden
test count has not been independently verified, so no full hidden-test runtime
is claimed. Only three visible studies were measured; larger/different DICOM
series may change preprocessing cost. The completed example therefore does
not guarantee hidden-test completion within the competition runtime limit.
