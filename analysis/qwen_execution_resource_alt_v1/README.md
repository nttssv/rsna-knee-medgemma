# Qwen execution resource alternatives v1

This directory contains a resource-only proposal and CPU-testable authorization gate for the same bounded Qwen3-14B smoke. It does not change model inputs, prompts, parsers, tokenization, inference settings, prepared files, or experiment order. It does not enable or run inference.

The proposed versioned allowlist contains exactly five 48 GB Secure Cloud GPU names: NVIDIA A40, NVIDIA RTX A6000, NVIDIA L40, NVIDIA L40S, and NVIDIA RTX 6000 Ada Generation. One GPU must be selected explicitly in a future private grant. The grant binds the chosen canonical GPU name, exact pod ID, exact region, and actual signed-in compute and storage rates. Runtime identity checks reject a GPU outside that list or any mismatch with the grant.

All five GPUs meet the architectural BF16 and CUDA 12.8 requirements in published specifications. The prior Qwen runtime recorded approximately 27.99 GiB peak GPU allocation. Each candidate has 48 GB; a future run must still pass the unchanged 36 GiB free-memory preflight. This is a compatibility assessment, not a tested model load or throughput guarantee.

The current public Secure Cloud rate snapshot and account-console capacity snapshot are recorded in [COMPARISON.md](COMPARISON.md). A selector result of “available” means the current Secure Cloud GPU picker showed the card for Any region with the selected PyTorch 2.8 template; it does not prove resumability of an existing pod or availability in the exact region the future grant selects.

The proposal remains **not authorized**. Its one-day quote-freshness window is a project rule, not a provider-issued quote expiration. The actual signed-in rate and storage configuration must be rechecked before a later approval. All candidates fit the existing $1.50 / 60-minute boundary using the proposal's $0.012/hour storage allowance, including L40S at an estimated $1.102 full-hour total.

Run the resource-only tests with:

```sh
pytest -q analysis/qwen_execution_resource_alt_v1/tests
```

The isolated `scripts/resource_alt_gate.py` is the required gate implementation for the future versioned runtime. The current runtime remains unchanged and retains its RTX 6000 Ada-only binding; it will reject these alternatives until a separately reviewed runtime version integrates this gate and binds the resulting source/plan hashes. Nothing in this directory authorizes a pod start or modifies `execution_enabled`.
