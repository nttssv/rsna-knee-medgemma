# T4 diagnostic: isolated CUDA attempts — 26 September 2026

Status: **FAILED_DUAL_GPU_OOM**. Cleanup repair verified by real Kaggle execution;
no example scores, submission or numerical pass. Source remains LOCAL_READY.

Private Kaggle version 6 (`353017107`) was requested at 22:35:08 SGT, with a
120-minute cap and no competition submission. Kaggle reports 205.7 seconds of
execution; this does not include all provider queue/startup time. Quota display
changed from `00:14 / 30 hrs` to `00:17 / 30 hrs` (displayed, rounded accounting).
At completion, version 6 was Failed, Active Events was 0 and Draft Session off.

## Observed memory (GiB)

| Attempt | Parent pre-attempt GPU0 / GPU1 | Immediate failure GPU0 / GPU1 | In-worker cleanup GPU0 / GPU1 | After worker exit GPU0 / GPU1 |
|---|---|---|---|---|
| single T4 | 14.460 / 14.460 | 4.184 / 14.360 | 10.321 / 14.360 | 14.460 / 14.460 |
| dual T4 | 14.460 / 14.460 | 4.811 / 13.714 | 10.936 / 14.345 | 14.460 / 14.460 |

Each child creates its own CUDA contexts; child pre-load free memory was
14.360 GiB/device versus 14.460 GiB in the parent. Both workers exited with
typed CUDA-OOM records and exit code 42 and were reaped. The first worker's
complete teardown restored both devices to the parent baseline. The existing
0.5-GiB recovery and 12-GiB free-memory gates passed before the dual worker.
The dual model load actually began at log time 171.0 s; both checkpoint shards
finished loading. The load/diagnostic stage then requested another 6.00 GiB on
GPU0 with only 4.81 GiB free. No third placement or retry was attempted.

| Attempt | GPU0 peak allocated / reserved | GPU1 peak allocated / reserved |
|---|---|---|
| single T4 | 12.705 / 13.148 | 0 / 0 |
| dual T4 | 12.382 / 12.520 | 0.321 / 0.631 |

Exact failing internal layer cannot be identified from the preserved wrapper
traceback; do not infer it from allocation size. No completed diagnostic means
there is no frozen successful `hf_device_map`/dtype report or No/Yes logit pair.
Cross-precision parity remains **NUMERICAL_PARITY_UNVERIFIED**.

Relevant terminal traceback:

```text
isolated_runner.py, run_isolated:
ContractError: two_gpu failed: ContractError: two_gpu CUDA OOM;
no further placement is permitted: CUDA out of memory.
Tried to allocate 6.00 GiB. GPU 0 ... 4.81 GiB is free ...; no fallback
```

## Environment and preserved experiment

Python 3.12.13, Torch 2.10.0+cu128, CUDA 12.8, two Tesla T4 GPUs.
All 32 offline assets verified; dependency installation took 17.41 s;
asset verification took 59.18 s. Pinned package checks passed.
Internet was off; notebook and model assets remained private.

Model ID/revision, NF4/double quantization/FP16, eager attention, LoRA,
448-pixel preprocessing, slices/series selection, target definitions,
No/Yes scoring and device-map policy were unchanged. AST comparisons verified
`ImageTeacher`, `Imaging`, `run_inference` and `validate_device_map` unchanged.
Asset manifest SHA256 remains
`4b4e9396301bd0c80a6a4ee1fd169e5b800d482f8833dcd28e3cf0b92ab4e0d7`.

The repair uses fresh exec workers, one shared deadline, load/diagnostic-only
OOM fallback, matching worker PID/placement/result and parent VRAM verification.
A successful worker would retain its model for the entire example; no reload
or changed model computation is introduced. 36 local tests passed, including
15 real synthetic subprocess tests. Executed source matched local source.

15 output files were downloaded and SHA256-inventoried in the private local run
folder `state/kaggle_image_t4_v1/runs/20260926T143508Z/`. Detailed device memory,
original OOM strings, worker exit records, configuration and environment are
preserved there; the earlier version 5 outputs are also retained separately.

**Remaining blocker:** the unchanged dual-T4 configuration still exhausts
GPU0 during load/diagnostic. Cleanup is no longer blocking its first real test.
No device-map, precision, image-count/resolution, attention, model or prediction
change was made to bypass this new failure; no additional GPU run was started.
