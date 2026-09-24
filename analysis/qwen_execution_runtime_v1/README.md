# Qwen live execution runtime v1

This separately versioned runner is prepared for one fixed Qwen inference smoke and is disabled by default. It consumes the private, reviewed five-report package and the frozen `qwen_runtime_v1` tokenizer, parser, plan verifier, and `report_labeling_llm_v2` offline Qwen backend/cache audit. It does not edit those artifacts.

The committed policy has `execution_enabled: false`. The CLI also requires `--execute`, the exact execution-plan SHA, the exact prepared-plan SHA, a private approval record, the reviewed exact-pod stop preflight, a live verified watchdog, and the complete pinned model cache. Without a later separately reviewed source/policy change and user-issued approval record, it exits before accessing approval files, cache, tokenizer, or model.

The scope is Qwen/Qwen3-14B at revision `40c069824f4251a91eefaf281ebe4c544efd3e18`; five fixed development reports; `control-1`, `candidate-1`, `candidate-2`, `control-2`; five reports per run; at most 20 generations. Training, validation inference, full-40 inference, bulk extraction, retries, repair generations, repeat selection, replacement hosts, substitute GPUs, and deadline/budget extensions are prohibited.

See [PROTOCOL.md](PROTOCOL.md) for the authorization schema, safeguards, and operator checklist. `execution_plan.json` is the exact stage plan; its SHA-256 must be provided to the runner and future approval record. The source has no API keys, credentials, approval record, weights, or cache.
