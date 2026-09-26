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
| Decoder numerical policy | BF16 residual stream | FP32 post-attention/post-MLP normalization results and residual additions; normalized branch inputs checked then cast FP16; NF4 compute FP16 verified after diagnostic |
| Vision image scheduling | All study images in one vision batch | One image per SigLIP call, concatenate all features in order before the unchanged projector; study batch remains 1 |
| Placement | One GPU, device 0 | Try GPU 0; after CUDA OOM, one explicit 2-GPU Accelerate map |
| Dual-GPU handling | Not supported | 7 GiB mapping budget per T4; no decoder/vision block splits; no CPU/disk offload |
| Inputs | BF16 floating tensors, integer IDs/masks | FP16 floating tensors on embedding device, integer IDs/masks unchanged |
| Score | FP32 softmax over No/Yes logits | Same; reject nonfinite native logits before FP32 conversion |
| Failure handling | Stop; never emit neutral 0.5 | Same; preserve per-study partials and placement/OOM log |
| Memory policy | One GPU ≥37 GiB, native BF16 | T4 ≥14 GiB each; ≥12 GiB free before load, ≥1 GiB free after diagnostic |
| Pre-run numerical check | Hardware + asset gate | Repeated same-input ACL logits/score check; thresholds 0.02 logit / 0.001 score |

The T4 adaptation changes precision and may change GPU placement. It does not
change the base checkpoint, adapter, processor, tokenizer, prompt definitions,
series/slice preprocessing, output schema or score definition. Versions 5/6
measured OOM and memory recovery; neither completed numerical diagnostics or
example inference. See [version 6 evidence](DIAGNOSTIC_2026-09-26_ISOLATED.md).

Version 7 keeps the existing placement policy. The frozen
processor resizes rendered 448px slices to 896px (unchanged), yielding 4096
SigLIP patch tokens. Six images × 16 heads × 4096² × FP32 is 6 GiB for one
eager-softmax tensor, consistent with the observed allocation request. This
source-based attribution is not an original internal stack trace from version 6.
The new inference-only vision wrapper retains the loaded tower, parameters and
Accelerate hooks and runs each image separately, reducing that tensor to 1 GiB.
All image features are concatenated in source order and the projector runs once.
No attention implementation, precision, image, prompt, checkpoint or score is
changed. Actual FP16 behavior remains subject to the repeated-input diagnostic;
BF16 numerical parity remains unverified. Original tracebacks, input shapes,
dtype reports and placement maps are now saved even when diagnostics fail.

The real [version 7 diagnostic](DIAGNOSTIC_2026-09-26_VISION_MICROBATCH.md)
removed the initial OOM on one T4, but failed the native finite-logit check.
Version 7 did not establish a complete example submission or BF16 parity.

The following version targets numerical overflow: 68 decoder post-normalization
modules call their original RMSNorm with FP32 input, preserving the existing
FP32 calculation instead of narrowing its output back to FP16. The two residual
additions consequently remain FP32. The 68 pre-normalization modules retain
finite, range-checked FP16 branch inputs to the existing attention/MLP modules.
Original norm parameters, LoRA and dispatch hooks remain intact. The actual NF4
compute dtypes are verified after the repeated diagnostic. No clipping, score
repair, model-wide conversion or fallback is allowed.

Local audit found all 883 base tensors and 272 adapter tensors finite and
representable in FP16 (base maximum absolute weight 1376; adapter 0.022903).
With the actual layer-33 post-MLP norm weight and a synthetic 2560-dimensional
input, the original FP16 narrowing returns Inf; FP32 returns 69671.296875,
above FP16's 65504 limit. This reproduces a mathematical hazard, not the actual
first failing operation in version 7. The first new diagnostic forward traces
module outputs (including decoder residual outputs), aborts on the first
nonfinite activation, and saves trace metadata even on failure. Attention masks
are not treated as activations; legitimate mask -Inf is not flagged.

Reference implementation: pinned transformers 4.57.6
[Gemma3 RMSNorm and decoder](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/gemma3/modeling_gemma3.py).
All dependency versions remain pinned; this patch does not use the current
Transformers main branch or alter the installed vendor source.

Version 8 completed all three example studies with this numerical policy.
Its first-forward trace found 29 finite module outputs above FP16's 65504
range: the first was decoder layer 5 (70858.1875), and the maximum was decoder
layer 29 (281899.8125), both FP32. This demonstrates that the observed residual
stream needs greater range; it does not retrospectively identify the exact
first failing operation of version 7. All 400 NF4 modules reported FP16 compute,
and repeated diagnostic logits/scores matched exactly in this one repeat.
See [the executed result](DIAGNOSTIC_2026-09-26_FP32_RESIDUAL.md).
