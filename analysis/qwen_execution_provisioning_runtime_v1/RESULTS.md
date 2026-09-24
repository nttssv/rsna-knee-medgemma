# Local controller validation

This package contains software tests and an unexecuted operational design. It has no measured GPU, model, performance, billing, or live shutdown result.

The controller enforces the immutable outer clock, exact A40 allocation, source/image/command bindings, one-create/one-resume limits, exact physical-machine identity, and cumulative-budget integration in CPU tests with fabricated adapters. It reuses the frozen inner grant/observation validators and source-integrity checks without editing those files. Both prompts/parsers, five reports, ABBA20 recipe, prepared tokens, and the 40/18 split remain unchanged.

The core suite verifies creation followed by immediate independent stop verification, one later resume with the same outer deadline, exact resource/quote/approval failures, owner-only files, exclusive durable receipts, cost type checks, bounded-count shutdown retries, and the unconditional live CLI rejection. Independent adversarial tests cover ambiguous responses, identity drift, stale observations, guard failure, dispatch-time freshness, cumulative costs, and loss of receipt durability.

Local validation: **77 tests passed** (43 core tests and 34 independent adversarial cases). The fresh-observation test includes nonzero observer latency; dispatch freshness still rejects excessive delay. These are CPU/synthetic tests only. The repository CI explicitly runs the focused Qwen/provisioning suites in addition to its baseline checks.

Run the current suite:

```bash
.venv/bin/python -m pytest -q analysis/qwen_execution_provisioning_runtime_v1/tests
```

**Execution remains disabled.** No production provider adapter, external shutdown worker, hard network-call supervision or inference handoff is implemented. Passing CPU tests cannot establish live provider control or capacity. The concrete remaining work is listed in [PROTOCOL.md](PROTOCOL.md#unresolved-live-prerequisites). This source milestone creates no real approval, reads no control credential and performs no provider mutation.
