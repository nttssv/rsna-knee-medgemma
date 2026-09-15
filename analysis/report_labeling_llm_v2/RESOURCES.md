# Proposed bounded GPU smoke — not authorized or executed

The original proposal below concerns generation. A subsequent [load-only preflight proposal](LOAD_PREFLIGHT.md) is separate. RunPod status and the current start quote have now been inspected; no GPU was started. The [local adapter](RUNTIME.md) is implemented, with execution locked and offline-only loaders. Its fake-backend tests do not establish real tokenizer or CUDA compatibility. Complete a reviewed offline five-report tokenizer preflight and runtime check before considering a paid stage.

Keep the v1 checkpoint IDs/revisions, BF16, batch size one, greedy decoding, one beam, SDPA, seed 20260914, 8,192 input-token limit and 2,048 new-token cap. See [MedGemma config](configs/medgemma.json) and [Qwen config](configs/qwen.json). No adapters, quantization or images are planned. Pinned tokenizer/config assets have been downloaded and verified; no v2 model weights have been downloaded. These are inherited recipes, not new measurements.

- MedGemma: `google/medgemma-1.5-4b-it`, revision `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b`.
- Qwen: `Qwen/Qwen3-14B`, revision `40c069824f4251a91eefaf281ebe4c544efd3e18`.

Qwen's future template should retain non-thinking mode. The v1 MedGemma runtime did **not** pass the similarly named config flag to its template, and reasoning-like output was observed; v2 does not claim thinking is disabled. Supported MedGemma template/output controls and EOS handling remain a runtime-review question. Do not introduce an unverified control or assume prompt wording prevents reasoning/truncation.

## Scope and cost proposal

Use the same five development reports, twice/model: 20 first generations total, no generated repair. Run models sequentially. Do not include validation report tokenization/inference, a full 40 run, MRI training or the other 4,349 reports. Save raw text, completion status, exact rendered inputs, normalization metadata, source spans, model/generation revisions, timestamps/tokens and GPU allocation. Transfer verified results locally and stop compute before any semantic review.

The historical A100 80GB v1 first-pass rates were approximately 72 seconds/report for MedGemma and 24 for Qwen: about 16 minutes of generation for 20 reports, excluding loading/setup. New prompts, completion behavior and evidence length can change this substantially. A **30–45 minute planning range** with a **one-hour proposed cap** is reasonable for another engineering smoke, not a guarantee. V1 peak PyTorch allocation was 8.65/27.99 GiB; those are not measured v2 requirements.

At the **historical 2026-09-14** RunPod quote of $1.59/hour, 30–45 minutes would be roughly $0.80–$1.19 compute, with a proposed one-hour limit of $1.59, excluding storage. Current availability and price must be checked before a paid stage. No live quote was obtained during this local task. Do not start a server based on this document.

Recommendation: a new five-case smoke is worth considering **after** the local protocol/test review and runtime integration questions are resolved. An external stop/monitor plan is needed to enforce a billing window; application generation timeouts alone do not stop provider billing. The intended teacher receives no relaxed technical or semantic gate.
