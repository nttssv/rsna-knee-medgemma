# Runtime diff from image baseline v1

Scientific inputs and output meaning remain the same. This runtime only adapts
execution for Kaggle T4 hardware.

| Area | `kaggle_image_baseline_v1` | `kaggle_image_t4_v1` |
|---|---|---|
| 4-bit weights | NF4 + double quantization | Same |
| Compute dtype | BF16 | FP16 for T4 |
| Unquantized base load | BF16 | FP16 |
| Adapter file | Frozen FP32 LoRA | Same file; actual dtype/device logged |
| Attention | Eager | Eager |
| Placement | One GPU, device 0 | Try GPU 0; after CUDA OOM, one explicit 2-GPU Accelerate map |
| Dual-GPU handling | Not supported | 7 GiB mapping budget per T4; no decoder/vision block splits; no CPU/disk offload |
| Inputs | BF16 floating tensors, integer IDs/masks | FP16 floating tensors on embedding device, integer IDs/masks unchanged |
| Score | FP32 softmax over No/Yes logits | Same; reject nonfinite native logits before FP32 conversion |
| Failure handling | Stop; never emit neutral 0.5 | Same; preserve per-study partials and placement/OOM log |
| Memory policy | One GPU ≥37 GiB, native BF16 | T4 ≥14 GiB each; ≥12 GiB free before load, ≥1 GiB free after diagnostic |
| Pre-run numerical check | Hardware + asset gate | Repeated same-input ACL logits/score check; thresholds 0.02 logit / 0.001 score |

The T4 adaptation changes precision and may change GPU placement. It does not
change the base checkpoint, adapter, processor, tokenizer, prompt definitions,
series/slice preprocessing, output schema or score definition. No T4 logits,
peak memory or example submission have been measured yet.
