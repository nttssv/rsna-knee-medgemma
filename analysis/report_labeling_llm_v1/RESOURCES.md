# Resources — measured smoke execution

The approved one-hour stage completed on 2026-09-14. The pod was confirmed **not running**, with compute **$0.00/hour**, after result transfer. Start attempt at 10:41:35 UTC to stop confirmation at 11:16:37 UTC was approximately 35 minutes. At the displayed $1.59/hour, this is approximately **$0.93 compute**, excluding storage and subject to actual provider billing. The initial machine was unavailable and the provider migrated the pod; no extra concurrent GPU was used.

## Exact checkpoints and recipe

| Item | MedGemma | Qwen |
|---|---|---|
| Model | `google/medgemma-1.5-4b-it` | `Qwen/Qwen3-14B` |
| Revision | `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b` | `40c069824f4251a91eefaf281ebe4c544efd3e18` |
| Published weight bytes | 8,600,277,880 | 29,536,665,640 |
| Input | Text-only official processor template | Text-only official template, thinking disabled |
| Precision | BF16, no quantization | BF16, no quantization |
| Observed peak PyTorch allocation | 8.65 GiB | 27.99 GiB |

Models ran sequentially with no adapter and no MRI input. The existing exact MedGemma snapshot was reused; Qwen was downloaded once. Model weights and Hugging Face caches are not committed. Publisher references: [MedGemma checkpoint](https://huggingface.co/google/medgemma-1.5-4b-it/tree/91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b), [Qwen checkpoint](https://huggingface.co/Qwen/Qwen3-14B/tree/40c069824f4251a91eefaf281ebe4c544efd3e18).

Exact [prompts](prompts/medgemma_prompt.txt), [Qwen prompt](prompts/qwen_prompt.txt), [condition definitions](prompts/target_definitions.txt), [MedGemma config](configs/medgemma.yaml), [Qwen config](configs/qwen.yaml) and [protocol](PROTOCOL.md) retain the executed fingerprints. Greedy decoding uses one beam, `do_sample=false`, seed 20260914, 2,048 maximum new tokens, 8,192 maximum input tokens and SDPA attention. No model setting was changed between repeats. This deterministic Qwen setting differs from the publisher's recommended sampling defaults.

All 58 full rendered inputs passed tokenizer preflight: maximum 1,781 MedGemma tokens and 2,600 Qwen tokens, zero overflow. The preflight includes validation report text for length checking only; no validation model generation or label evaluation took place. Exact templates, EOS/PAD values, resolved generation settings and rendered prompts are retained privately in tokenizer metadata and run records.

## Actual environment

- NVIDIA A100 80GB PCIe; `nvidia-smi` total 81,920 MiB and initially free 81,153 MiB. Driver 595.91.07, native BF16 supported. Allocated-memory peaks above exclude driver/reserved memory and are not total device usage.
- Replacement host configuration: 31 vCPU, AMD EPYC 7763, 117 GB host RAM, 30 GB container disk and existing 100 GB network volume. Earlier observations of a different host's 236 GB RAM do not describe this execution host.
- Python 3.12.3; PyTorch 2.8.0+cu128, CUDA runtime 12.8; Transformers 5.12.0, Hugging Face Hub 1.19.0, Accelerate 1.15.0, NumPy 2.5.3, pandas 3.0.5 and pytest 9.1.1. The resolved environment freeze is in the private execution snapshot.
- Isolated environment installed from [requirements-gpu.txt](requirements-gpu.txt). The saved MRI pilot environment was preserved. The template was `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`.
- Approximately 19 GiB was already in the workspace; Qwen cache added approximately 28 GiB and the isolated environment about 2 GiB. The smoke fitted the 100 GB volume without downloading MRI data. The network filesystem's `df` free-space figure reflects its backing filesystem, not the user's volume quota.

## Observed timing

| Run | Five-report generation time, including secondary | Mean first pass / report | Secondary generations | Command wall time, including loading |
|---|---:|---:|---:|---:|
| MedGemma 1 | 647.2 s | 72.4 s | 3 | 688.1 s |
| MedGemma 2 | 639.9 s | 71.5 s | 3 | 664.0 s |
| Qwen 1 | 113.5 s | 22.7 s | 0 | 206.6 s |
| Qwen 2 | 126.7 s | 25.3 s | 0 | 153.2 s |

Total generation time was approximately 25.5 minutes. Model commands took approximately 28.5 minutes including loading; setup, preflight, transfer and shutdown explain the rest of the approximately 35-minute stage. Qwen download overlapped MedGemma inference, so download time must not be added twice. Old planning throughput estimates are superseded by these measurements.

Linear extrapolation at unchanged output lengths gives roughly 86 minutes of MedGemma generation (including secondary passes) and 16 minutes of Qwen generation for one pass over 40 reports each, before loading/setup. **This is not an approved next run or a recommendation:** failures dominate the MedGemma workload, five reports are not a reliable runtime distribution, and any v2 recipe changes invalidate the extrapolation.

## Preservation and portability

The executed source commit is `f33495873f85d7b83480c7a65776c9cbea664fb5`. Fifty-four inference/preflight files were copied and verified by SHA-256. The result archive hash is in [smoke_execution.json](aggregate/smoke_execution.json). Four complete run manifests, tokenization artifacts, raw outputs, package freeze and controller timings are in private state. Provider identifiers, host addresses, SSH configuration and credentials are excluded from public artifacts.

Use the portable [review tools](smoke_review_tools/README.md) after restoring the private state backup. They require no GPU. A Git clone alone reproduces synthetic checks and public aggregates, not private case-level analysis. The original scripts/configs/protocol/tests are unchanged; postprocessing tools record their own provenance separately.
