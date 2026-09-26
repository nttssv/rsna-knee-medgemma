# Frozen V10 submission-only package

Scientific/runtime snapshot: **`26dafd65b91e054577110f4c695f7d72a7660faf`**.
The throughput optimization is complete. This package adapts dispatch and
packaging for hidden test IDs; it does not optimize the model further.
See [SUBMISSION_READY.md](SUBMISSION_READY.md) for checks, artifact hashes,
the exact scope of example parity evidence, and the execution boundary.

## Preserved inference

The notebook embeds the original V8 `inference_core.py`,
`submission_runtime.py`, `numerical_runtime.py`, configuration and requirements,
and the exact V10 `vision_cache.py`. These are checksum-bound, not copied into
new numerical implementations. The unchanged `ImageTeacher` constructor,
`score_logits`, `predict`, `run_inference`, cache and submission validator do
the actual work. Historical source files and outputs remain untouched.

- Same base revision, checkpoint, adapter and processor; original 32-file
  private offline asset package.
- NF4/double quantization, FP16 linear compute and existing FP32 residual/
  post-normalization policy; same FP32 LoRA behavior.
- Six sampled MRI images, same selection/order/resolution/preprocessing;
  same twelve target prompts and No/Yes FP32 softmax.
- Same per-study vision cache with exact pixel/context/weight checks,
  one miss/eleven hits, twelve scoring forwards and release on completion.
- Two independent T4 processes, each seeing one GPU and one full replica;
  no model parallelism, CPU/disk offload or resource fallback.
- Finiteness, completeness, source/asset checksums, worker-failure propagation,
  no-overwrite, deadlines and final schema checks remain mandatory.

## Removed diagnostic dependencies

The new runner has no V8 reference CSV, three-study ID list or parity gate.
It reads actual test IDs from `test.csv`, validates them, sorts them and assigns
alternating IDs to workers. Shard counts differ by at most one. Merge restores
the original `test.csv` order. Empty test sets fail; a one-study set has an
explicit empty shard that loads no second model. Neither failure nor an empty
shard invents predictions.

The direct frozen constructor replaces `prepare_t4_teacher`, whose diagnostic
performed two extra ACL forwards and an activation trace on the first visible
study. Those repeated comparisons/traces are omitted. Original per-forward
numerical checks, dtype/device checks, cache checks and final finite-score
checks remain. Loading failure stops the session; no alternate placement or
retry is attempted. There is no per-study <40-second acceptance gate: that
benchmark is complete and hidden cases may take longer without changing scores.

Only successful completion of both workers plus checksum/ID/schema validation
can publish `/kaggle/working/submission.csv`. Partial artifacts stay under the
run directory. The notebook does not invoke Kaggle submission APIs.

## Build locally

```bash
.venv/bin/python analysis/kaggle_image_t4_submission_v1/scripts/build_notebook.py \
  --output state/kaggle_image_t4_submission_v1/private_notebook/submission.ipynb
.venv/bin/python -m pytest -q analysis/kaggle_image_t4_submission_v1/tests
```

Use a new output path for another build; existing artifacts are not overwritten.
The generated notebook contains no saved predictions, study IDs, reports,
credentials or model weights. Metadata specifies Private and Internet OFF.
Attach the existing **RSNA Knee Image Assets v1 Private** dataset and the
**RSNA Knee Abnormality Detection** competition in Kaggle, select **T4 x2**,
and verify visibility/network settings before any authorized execution. The
asset files, base revision and dependencies are not downloaded at inference.

The future scored-execution timeout is **530 minutes from the first setup
cell**, leaving ten minutes within the competition's nine-hour ceiling. It
does not reset after setup or model loading and is not an authorization to
consume quota now. Parent supervision kills/reaps both worker groups on timeout
or worker failure. Completed execution must be checked for GPU/session shutdown
externally, as in V10.

## Local replay, not another model run

```bash
.venv/bin/python analysis/kaggle_image_t4_submission_v1/scripts/verify_v10_replay.py \
  --v10-dir state/kaggle_image_t4_vision_cache_v1/runs/20260926T163753Z/outputs/t4-vision-cache-20260926T164417Z \
  --output-dir state/kaggle_image_t4_submission_v1/v10_replay
```

This private, offline check sends saved V10 worker scores through the exact
new merge/validator and demands byte-identical final CSV output. It verifies
all 36 saved scoring records and their finite native No/Yes logits. It is not
embedded in the notebook and cannot constrain hidden IDs. It proves assembly
parity using real saved V10 results; it is explicitly **not fresh GPU inference**.
The unchanged inference functions/cache hashes and CPU dispatch tests provide
the separate source-level behavioral guarantee. No organizer labels are read.
