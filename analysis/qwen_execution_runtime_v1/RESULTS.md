# Results: Qwen live execution runtime v1

Implementation status: CPU/synthetic test stage only; no live inference has occurred. The real execution switch is off in `configs/runtime.json`. No RunPod resource was created, resumed, started, stopped, or charged during implementation.

The exact execution-plan SHA-256 is `0223e5be543e0bb8f4493302476ade6cf53b5ee08653acf42858bd8263e249c5`. It binds the prepared input-plan SHA-256 `c4586bb0c65163244dac80344ea96f691493a1b89a5911eefcba54f2384c74fc`, proposal SHA-256 `61d0dfe522f4ad4e4d00fef86532bc55f50c989a2495c6e9d6637be036165b12`, runtime policy/source, and frozen authorization-gate/watchdog source hashes.

The runtime requires a future exact-plan approval record, valid same-pod stop preflight, and verified live watchdog before checking the model cache or loading the tokenizer/model; it rechecks watchdog liveness before every generation and supervises the inference child with a hard kill/reap timeout. Results will be recorded under a private ignored `state/` directory only after a separately authorized live stage.

No accuracy, clinical, or generalization result is claimed by this implementation. The experiment is limited to five development reports and 20 generations; it does not use the 18-study validation split.
