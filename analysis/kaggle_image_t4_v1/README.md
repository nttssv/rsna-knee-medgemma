# Kaggle image inference adaptation for T4

This is a separate inference-only adaptation based on snapshot
`007b344b56d7c3fe1956ca913819c6bbcd19f079`. The frozen baseline package
`analysis/kaggle_image_baseline_v1/` and the MRI pilot checkpoint are unchanged.

Current state: **LOCAL_READY**; real Kaggle version 6 verified the process
cleanup repair and then failed with a genuine dual-T4 OOM. Both GPUs recovered
to 14.46 GiB after each worker exited. The dual attempt ran, but requested
another 6 GiB on GPU0 with 4.81 GiB free. No example predictions/submission.
See [the observed diagnostic](DIAGNOSTIC_2026-09-26_ISOLATED.md).

The repaired notebook isolates each placement in a fresh Python process. The
single attempt must exit and be reaped before parent-side synchronization and
VRAM checks. Only a matching CUDA-OOM result from load/diagnostic and exit code
42 permits one dual worker. Both GPUs still require 12 GiB free, and GPU 0 must
recover within 0.5 GiB of baseline. No model recipe, device map or score policy
changed. A successful worker completes the example without reloading its model.
An OOM during full example inference, non-OOM error, missing result, timeout or
identity mismatch stops the run without fallback. The first notebook cell sets
one shared 115-minute worker deadline (including setup), leaving five minutes
inside the user-approved 120-minute session cap for artifact collection/shutdown.

## Frozen image task

The runtime reuses the audited `google/medgemma-1.5-4b-it` base at revision
`91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b`, the same rank-8 adapter, processor,
tokenizer, 12 target definitions and order, No/Yes token IDs, prompts, series
selection, orientation ordering, two slices per plane, 448 px preprocessing,
and Yes-versus-No softmax score. DICOM failures stop the run. The submission
validator and no-neutral-score rule are retained.

The source does not rewrite any base or adapter weight. The baseline asset
manifest remains the same 32-file contract, SHA-256
`4b4e9396301bd0c80a6a4ee1fd169e5b800d482f8833dcd28e3cf0b92ab4e0d7`.
The adapter weight SHA-256 remains
`a0bbce8fade7c273dbbeeba39bf792f82eb166abf1bac3450d5100027496f448`.

## T4 runtime changes

- NF4 4-bit quantization and double quantization remain enabled.
- The quantized linear compute dtype and unquantized base model load dtype are
  FP16. The original base Safetensors are untouched.
- The adapter is loaded from the original FP32 adapter file and its actual
  dtype/device are recorded. No generic `.half()` or `.to("cuda:0")` is applied
  to an already quantized/distributed model.
- Floating processor tensors are moved to the input embedding device as FP16;
  integer token IDs and masks keep their integer dtype. Accelerate dispatch
  moves activations across module devices.
- Eager attention is retained. The model is placed in eval and inference mode;
  all parameters are frozen and no optimizer is created.
- Before loading, the runner accepts only one or two NVIDIA Tesla T4 devices,
  each with at least 14 GiB total VRAM and at least 12 GiB free on the device
  being tested. After the diagnostic forward it requires at least 1 GiB free on
  each participating GPU.
- It attempts one T4 first. Only a CUDA out-of-memory error permits one second
  attempt with an explicit Accelerate device map. The dual map uses 7 GiB
  placement budgets per device, respects Gemma3 decoder and SigLIP encoder
  no-split classes, requires modules on both GPUs, and rejects CPU or disk
  placement. After a single-T4 OOM, failure metadata is captured without
  retaining the exception traceback; failed model references are released,
  Python/CUDA caches are collected, all visible GPUs are synchronized, and
  free VRAM is remeasured on both devices. GPU 0 must recover within 0.5 GiB
  of its pre-attempt baseline and both GPUs must still pass the unchanged
  12-GiB gate before the dual-T4 constructor is entered. Otherwise the run
  stops with a cleanup-leak diagnostic. The attempt log records pre-failure,
  immediate post-failure and post-cleanup VRAM, plus whether each model load
  actually started. Actual `hf_device_map`, per-module dtype/device inventory,
  peak allocated/reserved memory per GPU and any failed attempt are saved.
- The selected placement is recorded and hashed after a repeated-input
  diagnostic passes, before the full visible example is scored.

Numerical checks inspect native selected No/Yes logits for NaN/Inf before
casting to FP32. The softmax is computed in FP32. No clipping or `nan_to_num` is
used. The ACL smoke input is run twice; maximum absolute logit change must be at
most 0.02 and the Yes-score change at most 0.001. A full BF16 reference on the
same example test images is unavailable, so cross-precision parity remains
`NUMERICAL_PARITY_UNVERIFIED`.

## Offline Kaggle package

The private notebook is generated locally at
`state/kaggle_image_t4_v1/private_notebook/rsna_knee_medgemma_image_t4.ipynb`.
The original, executed notebook SHA-256 is
`413d598f1742e97103fc412c5bccd2c2808d866a2067c3449db5a9c02c02f30a`.
The corrected local notebook is preserved separately as
`state/kaggle_image_t4_v1/private_notebook/rsna_knee_medgemma_image_t4_envfix.ipynb`,
SHA-256 `766759b1e5194a8749df1c77d2400bd5d2475351154f1c8aeead3c3d31c77d1d`.
It inventories every configured package pin and writes observed environment,
free memory and setup timings before running the unchanged validators.
The model asset folder is the existing ignored local package
`state/kaggle_image_baseline_v1/assets/`; the notebook expects one attached
private Kaggle input containing the asset manifest, base, adapter/processor and
wheelhouse. All manifest entries are streamed through SHA-256 verification
before install/load. Internet is off, packages install with `--no-index`, and
model/processor loading is local only. The wheelhouse supports CPython 3.11 and
3.12. The notebook records the actual Kaggle Python, Torch, CUDA and pinned
package versions, then fails before model loading if any package pin or CUDA
12.8 compatibility check differs.

The official bitsandbytes compatibility documentation lists T4/SM75 as
supported for 4-bit NF4 and CUDA 12.8 wheels as including SM75. This establishes
wheel/hardware compatibility, not that this model will fit or run correctly;
the Kaggle diagnostic and measured forward pass are still required.

## Output and timing

The notebook records separate dependency-install, asset verification, model
load, DICOM preprocessing and model-forward times; per-study timing; per-GPU
allocated/reserved peaks; numerical diagnostics; and `hf_device_map`. It writes
per-study partial rows as `submission.csv.partial`, but writes final
`submission.csv` only if every actual `test.csv` ID has all twelve finite
scores in `[0,1]` with the exact `sample_submission.csv` schema.

The local workspace has the three-row visible `test.csv` and matching series
metadata, but does not have the `test_series/<StudyInstanceUID>/...` DICOM
files. No local example inference can run. Kaggle test DICOMs are available
only after the private notebook is attached to the competition data.

Because one study requires twelve forwards, a three-study visible example
requires 36 prediction forwards plus the two repeated ACL diagnostic forwards.
No measured T4 throughput exists yet. The hidden-test runtime estimate and
headroom must be computed from the completed example study times and a verified
hidden study count; do not extrapolate from one forward. A successful example
does not prove the hidden run will finish under Kaggle's runtime limit.

## Build and checks

```bash
.venv/bin/python analysis/kaggle_image_t4_v1/scripts/build_notebook.py \
  --output state/kaggle_image_t4_v1/private_notebook/rsna_knee_medgemma_image_t4.ipynb

.venv/bin/python -m pytest -q analysis/kaggle_image_t4_v1/tests
```

Required private Kaggle inputs are the competition data and the existing
`state/kaggle_image_baseline_v1/assets/` directory packaged as one private
dataset. The notebook/model package must not be made public.

## Status definitions

- **LOCAL_READY**: adaptation, private notebook, unchanged asset checksums and
  CPU contract checks are complete.
- **KAGGLE_RUN_PASS**: a real Kaggle T4 run completed all example IDs and passed
  schema/numerical checks.
- **SUBMITTED** and **SCORED** require those actual external events.

Only **LOCAL_READY** is currently achieved. The user requested repair and
continuation after version 5, retaining the 120-minute bound. A failed prior
version remains preserved; only an actual complete example can establish
**KAGGLE_RUN_PASS**. No competition submission is authorized for this diagnostic.
