# Resource proposal — no model inference yet

Metadata checked 2026-09-14 against the publishers' Hugging Face repositories. No weights or tokenizers were downloaded. Models will run sequentially, one at a time, in BF16 without quantization or adapters.

| Item | MedGemma | Qwen |
|---|---|---|
| Checkpoint | `google/medgemma-1.5-4b-it` | `Qwen/Qwen3-14B` |
| Revision | `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b` | `40c069824f4251a91eefaf281ebe4c544efd3e18` |
| Parameters | 4,300,079,472 | 14,768,307,200 |
| Published weight bytes | 8,600,277,880 | 29,536,665,640 |
| Estimated peak GPU memory | 12–18 GiB | 36–44 GiB |
| Input mode | Text only, base instruction checkpoint | Text only, thinking disabled |
| Access | Gated; existing user access must be available on the server | Ungated |

The model IDs, revisions and weight sizes are observed metadata. Memory and time values are planning estimates, not measurements. MedGemma supports text input despite being multimodal. Qwen's official template supports disabling thinking; greedy decoding here differs from its recommended non-thinking sampling defaults. Sources: [MedGemma model card](https://huggingface.co/google/medgemma-1.5-4b-it), [Qwen model card](https://huggingface.co/Qwen/Qwen3-14B).

## Machine evidence

Local preparation used macOS arm64, Python 3.12.13, PyTorch 2.8.0, MPS available, **no CUDA GPU**. Approximately 52 GiB was free during the local audit. Transformers, Accelerate, bitsandbytes and Hugging Face Hub were absent from that environment; the configured model cache was absent. No GPU model memory, inference rate or context-token measurement was obtained locally.

The authenticated RunPod console showed the migration pod **not running**, compute **$0.00/hour**, a restart price of **$1.59/hour**, A100 PCIe configuration, 236 GB host memory, 30 GB container disk and a 100 GB network volume. Its configured image identifies PyTorch 2.8.0 / CUDA 12.8.1. The preserved pilot records identify the GPU as A100 80 GB. Actual currently available VRAM, NVIDIA driver, CUDA runtime, installed packages, model-cache contents and free network-volume space cannot be measured while the pod is stopped. Configured image versions are not a live environment audit.

## Disk and time estimates

Both complete weight sets total **38.14 GB (35.52 GiB)**. Allow roughly **45–50 GiB free on persistent storage** for weights, tokenizers, environment packages, temporary files and analysis outputs. Do not rely on the 30 GB container disk for both models. A 100 GB network volume can be sufficient if that much space is free; this proposal does not require the approximately 570 GB MRI dataset. Reuse an existing verified MedGemma snapshot if available. Do not delete existing data or weights to make room automatically.

Illustrative per-model estimate for 58 reports at 1,000 output tokens/report: MedGemma at an assumed 50–100 tokens/s takes about 10–20 minutes of decoding; Qwen at an assumed 20–50 tokens/s takes about 20–49 minutes. These are unmeasured assumptions, excluding prompt prefill, model loading, second passes and downloads. Both full runs plus repeated development smoke tests and setup might fit a **1–2 hour planning window**, but long outputs, repair attempts or slow downloads can exceed it. At the observed $1.59/hour, that window corresponds to **$1.59–$3.18 compute**, excluding existing storage charges. This is not a spending authorization or hard cost guarantee.

The proposed first paid stage is limited to a hardware/disk audit, tokenizer preflight and **five development studies per model repeated twice**. Request approval for that bounded smoke stage first, then replace estimates with measured throughput and peak VRAM before full development/validation. No script provisions a pod, adds credit or stops provider billing.

Use an isolated environment with the pinned `requirements-gpu.txt`; the preserved MRI environment remains unchanged. Publisher APIs confirmed the specified Transformers 5.12.0 and Hub 1.19.0 releases exist. Actual Linux dependency resolution and CUDA/model-loading compatibility are still unverified and must be checked during the approved smoke stage.
