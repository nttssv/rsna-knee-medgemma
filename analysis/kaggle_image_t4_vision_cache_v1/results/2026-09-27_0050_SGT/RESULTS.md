# Same-study vision cache: real T4 result below 40 seconds

**UNDER40_PARITY_PASS — 2026-09-27, Asia/Singapore.**
Private [Kaggle Version 10, script 353048812](https://www.kaggle.com/code/tungsingapore/rsna-knee-image-baseline-t4-inference-private/log?scriptVersionId=353048812)
completed all three visible test studies in **23.570, 25.802 and 23.655 seconds**
per complete study. Every study included image preprocessing and all twelve
target forwards. All **36 worker scores exactly match Version 8**, with maximum
absolute difference **0**. This satisfies the predeclared strict <40-second
target on this example set; it does not establish a hidden-test worst case.

Execution source: `36fa04c724a2bc6488254f743b2bcb03b60dd46e`.
Frozen scientific snapshot: `4a4382ab8546cd872e90a3e81654291be9fa7546`.
Save & Run was requested at **2026-09-27 00:39:52.918 SGT**, with an external
30-minute ceiling starting at that request. Kaggle measured **253.3 seconds**
of notebook execution. Queue/bootstrap and UI reporting delay are not part of
that execution figure. No second run, retry or competition submission occurred.

## Measured complete-study performance

| Measurement | GPU0 / worker 0 | GPU1 / worker 1 |
|---|---:|---:|
| Studies completed | 2 | 1 |
| Complete-study seconds, shard order | **23.570; 25.802** | **23.655** |
| Version 9 same-study seconds | 93.070; 99.074 | 95.740 |
| Model load | 45.282 s | 45.305 s |
| Study loop wall time | 49.399 s | 23.681 s |
| Process completion after parallel launch | 122.101 s | 98.609 s |
| Peak allocated memory | 4.776 GiB | 4.776 GiB |
| Peak reserved memory | 5.662 GiB | 5.662 GiB |
| Vision cache misses / hits | 2 / 22 | 1 / 11 |
| Underlying six-image vision executions | 2 | 1 |
| Vision CUDA-event time, all cache misses | 12.838 s | 6.214 s |
| Sampled GPU utilization, whole window mean / maximum | 49.83% / 100% | 31.52% / 100% |
| Sampled utilization before worker exit, mean | 49.83% | 38.59% |

The complete-study mean is **24.342 seconds**, versus **95.962 seconds** in
Version 9: approximately **3.94x faster**, or **74.6% less time**. Each study
passes individually; this is not a two-GPU average disguised as per-study latency.
Parent launch-to-merge/parity wall time was **122.132 s**, including imports,
loading and two original diagnostic forwards per worker. Dividing that small
sample's entire worker window by three gives **40.711 s/study including worker
setup**. Full notebook execution took **4 min 13.3 s**. Setup is reported
separately, not hidden in the under-40-second claim.

All 32 asset checksums were verified in **63.916 s**; offline dependency
installation took **17.741 s**. Utilization uses 60 `nvidia-smi` samples per
GPU, roughly two seconds apart, with zero sampling errors. Whole-window means
include model loading and GPU1's idle tail from the 2/1 split. Memory peaks
cover loading, diagnostic and inference. The frozen `inference_seconds`
accumulator includes diagnostic forwards; use `study_timings.csv` for complete
study performance, not that accumulator.

## Scores, numerical checks and frozen behavior

- **36/36 raw worker scores exact versus V8**; predeclared absolute tolerance
  `1e-6`, relative tolerance `0`, passed without changing it after the run.
- Final merged CSV has three rows, thirteen columns, correct test order,
  unique/complete IDs and 36 finite scores in [0,1]. Its SHA equals the V9
  merged CSV. The pandas CSV round-trip differs from V8 by at most **9e-17**
  in decimal comparison (**9.71445146547012e-17** as Python floats); it is not
  claimed byte-identical to V8. Raw worker values are exact.
- Native No/Yes logits and FP32 softmax are finite. Repeated-input logit and
  score deltas are zero on both workers; no nonfinite activation was observed
  in the initial traced forwards. All 36 scored forward records are retained.
- One miss and eleven hits per study, twelve scoring forwards, and cache
  release after completion were verified. No cross-study reuse occurred.
- No DICOM errors, fallback values, clipping, repair, retries or label-based
  selection occurred. Cross-precision BF16 parity remains
  **NUMERICAL_PARITY_UNVERIFIED**.

Only the output of the unchanged vision encoder is reused for identical pixel
bytes within one study. Each target still executes the unchanged projector and
decoder. Checkpoint, adapter, image order/count, 448px source images, processor
(actual tensor `[6,3,896,896]`), eager attention, target prompts and No/Yes
score computation are unchanged. Original V8 NF4/double-quantization and
FP16 compute remain, together with V8's existing FP32 post-normalization/
residual policy and FP32 LoRA. Integer inputs remain int64.

Each worker sees exactly one Tesla T4 via `CUDA_VISIBLE_DEVICES=0` or `1`;
both consequently record local `hf_device_map={"":"0"}`. There is no model
parallelism or CPU/disk offload. Actual environment: Python 3.12.13,
Torch 2.10.0+cu128, CUDA 12.8, transformers 4.57.6 and the unchanged pinned
packages recorded in [aggregate metrics](metrics.json).

## Integrity, quota and shutdown

| Artifact | SHA-256 |
|---|---|
| Executed source manifest at `36fa04c` | `455f9e95210845ed51f471e37a73e2bc4b56d358ccff1c70e20f93047737564a` |
| Uploaded private notebook | `d957a617f8557a29323c8bc6ded917bef81e24a3ccd828b7b0d51d54f4ea411b` |
| Original 32-file asset manifest | `4b4e9396301bd0c80a6a4ee1fd169e5b800d482f8833dcd28e3cf0b92ab4e0d7` |
| Original LoRA adapter | `a0bbce8fade7c273dbbeeba39bf792f82eb166abf1bac3450d5100027496f448` |
| V8 reference CSV | `1f3d823c4b1e689b252c041945440dcc3685d0e5810cb7fbd91583739ef739cd` |
| Final example CSV | `8b66aef38061782800d830dd1c4bce4e21024a0ea34c6e404c98b9bafdb1a7ec` |
| Downloaded output ZIP | `dfb61485e65adfe98bfc4c1bfa733bcb8252829662c7e3a2ad41bf20ec9e3e23` |

Base remains `google/medgemma-1.5-4b-it` at
`91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b`. Downloaded runtime/config
sources match the committed hashes. All **48 output files** were downloaded
and hashed locally, including worker logs, raw logits/scores, cache receipts,
timings, traces, dtype/device maps, parity and final example CSV. Private IDs
and predictions stay in ignored local state; this report publishes aggregates.

Kaggle quota changed from **00:40 to 00:44 / 30 hours**: a **4-minute displayed
delta** at minute resolution. This is the provider UI observation, not a
two-GPU multiplication or a claim of second-precise billing. Version 10 was
**Successful**, **Draft Session off**, and **0 Active Events** were observed.
Notebook and model assets stayed Private, Internet OFF. The one session
finished within both 30-minute limits. No competition submission was made.

Before the run, **218 focused CPU tests passed**: 69 cache/runner, 70 prior
replica, 79 frozen T4 tests. The exact execution-source CI passed. After the
run, independent local checks verified source hashes, all scores, schema/order,
timing scope, cache counts and finite numerical records. No historical result
or scientific/runtime source was changed when recording these measurements.

## Scope of readiness

**The <40-second example goal is achieved.** Historical V8 `KAGGLE_RUN_PASS`
and V9 `THROUGHPUT_READY` remain intact. Quality/medical accuracy was not tested
by this speed diagnostic.

For an **illustrative 1,300-study workload**, two balanced workers at the
slowest observed 25.802 s/study imply **4.66 h** before setup. Adding 20% and
ten minutes gives **5.76 h**. The count is a planning assumption here, and
three visible cases do not establish hidden-study size/IO distribution or
tail latency. The current notebook deliberately binds the three-study V8
reference gate and is **not a hidden-test submission notebook**. A submission
path must remove that diagnostic-only reference dependency without changing
inference and must be explicitly authorized before execution/submission.
