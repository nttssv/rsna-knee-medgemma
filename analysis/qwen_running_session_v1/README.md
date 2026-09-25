# Qwen running-pod extraction session v1

This is a prospective operator entry point for the single approved, already-running RunPod pod `low_brown_viper` (RTX A6000). It leaves all frozen Qwen science artifacts intact and does not use the historical stopped-pod/resume live gate.

The runtime is disabled by default. Its session policy binds the exact selected pod by SHA-256, Secure Cloud / EU-SE-1, one RTX A6000, 50 GB host RAM, 80 GB container disk, $0.53/h compute, $0.011/h storage, a three-hour T0-based maximum, $3 session cap, and a T+175-minute stop backstop. The prior approximately $0.41 from an earlier session is recorded separately.

Scientific scope is fixed to the existing five-report development package, pinned Qwen3-14B revision, frozen control/candidate prompts, ABBA order, BF16, SDPA, greedy batch-one generation, the 2,048-token output cap, and frozen v2/v1 parsers. This session performs report extraction only: no MRI training, MedGemma, fine-tuning, validation, bulk extraction, repair, or retries.

`configs/execution_plan.json` and `configs/source_manifest.json` bind this prospective operator source and the full protected historical source set. Run `python scripts/qwen_operator_run.py --help` for the command interface. The separate `external_stop_guard.py` accepts only the exact selected pod ID and makes at most one provider stop request. Its private credentials and session records belong under ignored `state/`, never in this directory or Git.

The provider observation path is the signed-in RunPod console and live pod terminal. Unknown provider facts remain blockers; a Python process exit is not provider shutdown. Model inference may begin only after pinned cache verification, token parity, hardware/package checks, and a usable external stop path.
