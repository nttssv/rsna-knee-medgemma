# Source milestone results

No GPU or provider resource was started, resumed, created, stopped, resized, or modified. No approval record was created. Real execution remains disabled in the committed runtime policy.

The alternate runtime source is implemented as a separate version and pins prepared-plan SHA-256 `c4586bb0c65163244dac80344ea96f691493a1b89a5911eefcba54f2384c74fc`, prior execution-plan SHA-256 `0223e5be543e0bb8f4493302476ade6cf53b5ee08653acf42858bd8263e249c5`, alternate proposal SHA-256 `884dc38d2004abb44c64d2363e525969bf7f7fabd35f86eb5d03144bf34e72e2`, and new execution-plan SHA-256 `ab82f51cf3883e93b2c33bf750a32148297226ebd3e9ada0febb54d18d85a4f5`.

The plan binds the alternate runtime policy, live source, resource gate, prior watchdog gate, and watchdog source. The live-source SHA-256 is `90bd957ea45b64eb42ef709d7f2c36211de635775055ebbb4d6f71160b6fa56c`; the runtime policy SHA-256 is `d290e19f875fcf51a9f096f6bc32093bca3e5ba113318240e091bc88569e1737`.

The provider-observation gate requires an exact resource match, a timezone-aware observation timestamp after approval and no more than 10 minutes before the requested start, and `provider_state: "STOPPED"`. That is pre-start evidence only; it is not treated as the pod's current state after start. The same-pod preflight and live watchdog cover the subsequent control path.

The fixed Qwen experiment remains unchanged: Qwen3-14B at its pinned revision with BF16, SDPA, batch one, five fixed development reports, four ABBA blocks, and at most 20 generations. No training, validation, full-development inference, bulk extraction, retries, or repair is introduced.

Validation: 132 CPU tests passed across the reviewed execution runtime, its watchdog/resource gate, the alternate proposal gate, and the new alternate runtime. The new runtime contributes 37 tests. Python compilation passed for the live runner and alternate resource gate. Markdown links and credential scan passed for the earlier source commit; they are rerun for this correction. Any later live run still needs separate spending approval and fresh exact-pod verification.
