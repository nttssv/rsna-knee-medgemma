# Source milestone results

No GPU or provider resource was started, resumed, created, stopped, resized, or modified. No approval record was created. Real execution remains disabled in the committed runtime policy.

The alternate runtime source is implemented as a separate version and pins prepared-plan SHA-256 `c4586bb0c65163244dac80344ea96f691493a1b89a5911eefcba54f2384c74fc`, prior execution-plan SHA-256 `0223e5be543e0bb8f4493302476ade6cf53b5ee08653acf42858bd8263e249c5`, alternate proposal SHA-256 `884dc38d2004abb44c64d2363e525969bf7f7fabd35f86eb5d03144bf34e72e2`, and new execution-plan SHA-256 `f5198d2d5684a60196cca4367f55344fe0ae8d4c67cef304a3e120bf4fd0d227`.

The execution plan binds the alternate runtime policy and source, the alternate resource gate, and the reviewed watchdog gate/source. Its source SHA-256 is `140021db791c59572575d225a054574857959eb71dd93d168933412e51741e82`; policy SHA-256 is `d290e19f875fcf51a9f096f6bc32093bca3e5ba113318240e091bc88569e1737`.

The runner accepts one exact GPU selection per future grant from the five-device versioned allowlist and applies per-GPU rates from the proposal. A private provider-observation receipt must match the approved pod, region, hardware, storage, and rates. Any mismatch fails closed; no fallback is implemented.

The fixed Qwen experiment remains unchanged: Qwen3-14B at its pinned revision with BF16, SDPA, batch one, five fixed development reports, four ABBA blocks, and at most 20 generations. No training, validation, full-development inference, bulk extraction, retries, or repair is introduced.

Validation: all 126 CPU tests passed across the reviewed execution runtime, its watchdog/resource gate, the alternate proposal gate, and the new alternate runtime. The new runtime contributes 31 tests. `py_compile` passed for the live runner and resource gate. A proposal/resource capacity snapshot is not a guarantee that any particular pod can start; any future live stage must verify the exact selected pod and get separate explicit spending approval. This implementation stage did neither.
