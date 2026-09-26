# Two independent T4 replicas: measured diagnostic

**THROUGHPUT_READY — 2026-09-27, Asia/Singapore.**
Private [Kaggle Version 9, script 353037882](https://www.kaggle.com/code/tungsingapore/rsna-knee-image-baseline-t4-inference-private/log?scriptVersionId=353037882)
completed all three visible test studies, with 36 finite model-derived scores.
No competition submission was made. The original Version 8 **KAGGLE_RUN_PASS**
and historical artifacts remain unchanged.

Execution source: `7b0c075441c836ea12150a8f7ea1a88065220028`.
Frozen scientific snapshot: `4a4382ab8546cd872e90a3e81654291be9fa7546`.
The Save & Run request was recorded at 2026-09-26 23:56:49 SGT;
Kaggle displayed the version creation at 23:56:55 SGT. Kaggle's measured
execution time was **391.4 seconds**; queue/bootstrap delay is not included
in that execution figure. The external ceiling was measured from the request,
not reset at model loading.

## Score parity and integrity

| Check | Observed result |
|---|---|
| Worker model scores versus V8 | **36/36 exact**, maximum absolute difference **0** |
| Parent's frozen pandas comparison | 36/36 exact, max difference 0; predeclared `atol=1e-6`, `rtol=0` passed |
| Final merged CSV, independent decimal comparison | 13/36 strings represent exactly the same decimal; maximum difference **9e-17** |
| Final merged CSV, Python float comparison | Maximum difference **9.71445146547012e-17** |
| Submission contract | 3 rows, 13 columns, exact original test order, no missing/duplicate IDs, 36 finite scores in [0,1] |
| Repeated diagnostic on each GPU | Native logits and score differences 0; no observed nonfinite activation |
| DICOM issues / fallback / retries | 0 / none / none |

The tiny merged-file difference is the pandas read/write decimal round-trip;
the downloaded raw per-worker predictions match the V8 CSV exactly. Neither
scores nor logits were clipped, repaired or selected. The final CSV is not
byte-identical to V8, and we do not claim it is. No labels were read for this
comparison. Cross-precision BF16 parity remains **NUMERICAL_PARITY_UNVERIFIED**.

The four frozen runtime/config files and requirements in the downloaded output
match the original source hashes. Each worker used a distinct process, one
visible T4 and one model load. `CUDA_VISIBLE_DEVICES=0` and `=1` select the two
physical GPUs; both workers consequently report local `hf_device_map={"":"0"}`.
No CPU/disk offload or model-parallel path was used.

## Actual timings and memory

| Measurement | GPU0 / worker 0 | GPU1 / worker 1 |
|---|---:|---:|
| Assigned studies | 2 | 1 |
| Model load | 42.641 s | 42.709 s |
| Complete-study times, shard order | 93.070 s; 99.074 s | 95.740 s |
| Complete-study mean | 96.072 s | 95.740 s |
| Study loop wall time | 192.171 s | 95.764 s |
| Process completion after parallel launch | 261.726 s | 167.029 s |
| Peak allocated memory | 4.776 GiB | 4.776 GiB |
| Peak reserved memory | 5.662 GiB | 5.662 GiB |
| Free VRAM before worker load | 14.460 GiB | 14.460 GiB |
| Sampled GPU utilization, whole parallel window mean / maximum | 77.53% / 100% | 42.49% / 100% |
| Sampled utilization before this worker exits | 77.53% | 66.33% |
| Process lifetime / parallel wall | 99.99% | 63.81% |
| Complete-study work / parallel wall | 73.41% | 36.58% |

GPU utilization is measured from 128 `nvidia-smi` samples per GPU at roughly
two-second intervals, with zero sampling errors. The unweighted sample mean
includes load and, for GPU1, its idle tail. It is not the process/work fraction.
The uneven 2/1 assignment explains GPU1's lower whole-window utilization.

Asset checksum verification took **66.333 s**, offline dependency installation
**17.309 s**. The parent measured **261.761 s** from parallel worker launch to
validated merge/parity, including imports, loading, two diagnostic forwards
per worker and study inference. Each assigned study still used exactly twelve
prediction forwards; four additional diagnostic forwards were recorded as
overhead. Complete-study timers, rather than individual forwards, underpin
the throughput estimate. The frozen `inference_seconds` accumulator also
includes diagnostic forwards and must not be treated as study-loop wall time.

| Comparison with Version 8 | V8 | Two replicas |
|---|---:|---:|
| Kaggle execution wall time | 625.5 s | **391.4 s** |
| Model-load wall time per worker | 43.337 s | 42.641 / 42.709 s |
| Full three-study serial loop / longest shard loop | 286.391 s | **192.171 s** |
| Asset verification | 197.188 s | 66.333 s |

The entire notebook was **1.60x faster (37.4% less execution time)**, but that
gain is not solely replica scheduling: asset verification was also faster on
this host. The serial loop versus longest two-study shard is approximately
1.49x; the three-study sample has unavoidable 2/1 load imbalance. Per-study
compute itself remained about 96 seconds.

## Frozen arithmetic and assets

Both workers loaded the same MedGemma base revision and LoRA adapter, with
400 quantized linear modules computing in FP16, FP16 vision and normalized
branch inputs, FP32 LoRA and the unchanged V8 FP32 post-normalization/residual
policy. Token IDs, masks and token-type IDs remained int64; pixel values
became FP16 at the model boundary. All six images were retained. The unchanged
processor produced `[6,3,896,896]` tensors from the existing 448px PIL images.
Eager attention, twelve target prompts and FP32 No/Yes softmax were unchanged.

- Model: `google/medgemma-1.5-4b-it`, revision `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b`.
- Adapter SHA-256: `a0bbce8fade7c273dbbeeba39bf792f82eb166abf1bac3450d5100027496f448`.
- Asset manifest SHA-256: `4b4e9396301bd0c80a6a4ee1fd169e5b800d482f8833dcd28e3cf0b92ab4e0d7`; all 32 files verified.
- Uploaded notebook SHA-256: `f686065dc81bbfe0745089cee5fcf47d1125064a43ab6c99b706f535675e9586`.
- Final CSV SHA-256: `8b66aef38061782800d830dd1c4bce4e21024a0ea34c6e404c98b9bafdb1a7ec`.
- Environment: Python 3.12.13, Torch 2.10.0+cu128, CUDA 12.8, two Tesla T4s.
- Frozen package pins: transformers 4.57.6, huggingface-hub 0.36.0, tokenizers 0.22.1, peft 0.20.0, accelerate 1.12.0, bitsandbytes 0.50.2, pydicom 3.0.2, python-gdcm 3.0.26, sentencepiece 0.2.1, safetensors 0.6.2.

## Quota, artifacts and remaining runtime limit

Kaggle's displayed quota changed from **00:33 to 00:40 / 30 hours**, a
**7-minute displayed delta**, at one-minute UI resolution. This is the
provider's observed quota delta, not a doubled estimate for two GPUs. The
diagnostic stayed inside the 120-minute wall/quota ceiling. Draft Session was
off and **0 Active Events** was independently observed after completion.
Notebook and attached model assets remained Private; Internet remained OFF.

All **42 output files** were downloaded and hashed locally, including both
worker logs, raw prediction files, per-study timings, activation traces,
dtype/device maps, utilization samples, parity detail and the final CSV.
Study IDs and per-study scores remain in ignored private state. This public
report and [aggregate metrics](metrics.json) contain no study identifiers.

Using the roughly **1,300 hidden studies** described by the
[competition data page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data),
650 studies per worker at the slower observed complete-study mean implies
**17.35 hours before setup overhead**. Adding 20% runtime margin and ten
minutes for startup/IO yields **20.98 hours**. This is an extrapolation from
only three visible studies; hidden series size and IO may differ. It still
exceeds the [competition's nine-hour notebook limit](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview).
The example-only reference gate also intentionally rejects other test IDs;
this private diagnostic notebook is not a hidden-test submission notebook.

The requested two-replica implementation and real parity diagnostic are
complete. **THROUGHPUT_READY does not mean hidden-test runtime ready.**
No model semantics were changed to force a faster result. No competition
submission, training, Qwen inference, RunPod resource or paid service was used.

## Validation

Before the external run: **149 CPU tests passed** (70 replica tests and
79 frozen T4 tests), including synthetic independent-process dispatch,
failure/timeout sibling cleanup, no-overwrite behavior, identity/shard checks,
frozen source hashes and all-36-score parity failures. After downloading,
independent checks confirmed raw score equality, merged schema/order,
exact worker isolation and unchanged arithmetic/source hashes. The tests
are software evidence; the measurements above are from the real Kaggle run.
