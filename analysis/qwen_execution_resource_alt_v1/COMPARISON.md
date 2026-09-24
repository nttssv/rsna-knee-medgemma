# GPU comparison

Public price source: [RunPod GPU pricing](https://www.runpod.io/pricing), updated 2026-09-13 and checked 2026-09-24. It lists the following Secure Cloud rates and 48 GB memory for each candidate. Storage is separately priced at $0.10/GB/month for container disk, or about $0.011/hour for 80 GB; the authorization ceiling remains $0.012/hour. Public rates are list prices; the signed-in configuration must be read again before any future approval.

The capacity column is an account-console snapshot taken 2026-09-24 at approximately 15:22 UTC. The GPU picker showed Cloud type **Secure**, **Any region**, and the RunPod PyTorch 2.8 template. “Available” is an aggregate selector result, not a promise for US-WA-1 or a pod-specific resume check.

| Exact NVIDIA/provider name | RunPod picker label | VRAM | Secure rate | Picker capacity | BF16 / CUDA 12.8 | Memory assessment | Runtime/model-loading impact |
|---|---|---:|---:|---|---|---|---|
| NVIDIA A40 | A40 | 48 GB | $0.49/hr | Available, Any region | Yes; CC 8.6 | Likely fit with unchanged ≥36 GiB free gate; prior peak was ~27.99 GiB | Ampere instead of Ada; same CUDA/Transformers path and SDPA setting; no recipe change identified |
| NVIDIA RTX A6000 | RTX A6000 | 48 GB | $0.53/hr | Available, Any region | Yes; CC 8.6 | Likely fit with unchanged ≥36 GiB free gate | Ampere instead of Ada; same CUDA/Transformers path and SDPA setting; no recipe change identified |
| NVIDIA L40 | L40 | 48 GB | $0.82/hr | Out of capacity, Any region | Yes; CC 8.9 | Likely fit with unchanged ≥36 GiB free gate | Ada; same CUDA/Transformers path and SDPA setting; no recipe change identified |
| NVIDIA L40S | L40S | 48 GB | $1.09/hr | Available, Any region | Yes; CC 8.9 | Likely fit with unchanged ≥36 GiB free gate | Ada; same CUDA/Transformers path and SDPA setting; no recipe change identified |
| NVIDIA RTX 6000 Ada Generation | RTX 6000 Ada | 48 GB | $0.84/hr | Available, Any region | Yes; CC 8.9 | Likely fit with unchanged ≥36 GiB free gate | Existing approved hardware; unchanged path |

RunPod's GPU listing provides the names, rates, memory, and per-model system memory/vCPU configuration. NVIDIA's [compute capability table](https://developer.nvidia.com/cuda/gpus) classifies A40 and RTX A6000 as CC 8.6, and L40, L40S, and RTX 6000 Ada as CC 8.9. NVIDIA documents BF16 for CC 8.0 and higher in the [CUDA programming guide](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/05-appendices/mathematical-functions.html), and marks Ampere and Ada as supported by ongoing CUDA toolkit releases in its [CUDA architecture matrix](https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html). The assessment of the existing PyTorch 2.8.0+cu128 build on CC 8.6/8.9 follows those architecture/toolkit support facts; it still requires the unchanged on-host CUDA, native BF16, and SDPA checks.

### Cost at the one-hour maximum

| GPU | Compute for 60 min | Storage allowance for 60 min | Estimated maximum-window total |
|---|---:|---:|---:|
| NVIDIA A40 | $0.490 | $0.012 | $0.502 |
| NVIDIA RTX A6000 | $0.530 | $0.012 | $0.542 |
| NVIDIA L40 | $0.820 | $0.012 | $0.832 |
| NVIDIA RTX 6000 Ada Generation | $0.840 | $0.012 | $0.852 |
| NVIDIA L40S | $1.090 | $0.012 | $1.102 |

These estimates are below the unchanged $1.50 cap. Actual billing depends on the signed-in rate and verified resource state; the provider does not enforce this project budget ceiling.

### Interpretation and limits

All candidates have 48 GB (RunPod's listing); the prior ~27.99 GiB peak leaves theoretical headroom, but allocation measurements from a different GPU are not a guarantee. The existing ≥36 GiB free-memory preflight remains mandatory. A40 and RTX A6000 use Ampere rather than Ada; their model-loading and kernel execution remain within the same generic CUDA/PyTorch stack, but exact load and SDPA behavior must pass runtime checks. Throughput on these five cards has not been measured for this frozen run. The runtime's 1,800-second hard inference supervisor remains the limit; if it expires, the run fails without a retry. Hardware may cause small numerical differences, so results from different GPU types should be labeled with the exact approved device and not treated as bit-for-bit cross-device replications.
