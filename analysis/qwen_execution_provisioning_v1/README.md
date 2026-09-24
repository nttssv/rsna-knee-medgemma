# Qwen provisioning proposal v1

Status: **local documentary proposal; implementation and review outstanding**. Provisioning, resume, inference, and spending switches are all false. This directory contains no provider client or executable provisioning path and creates no approval record.

A new RunPod pod starts billable compute during creation. The [reviewed alternate-GPU runner](../qwen_execution_runtime_alt_v1/PROTOCOL.md) instead requires an exact existing pod's stopped-state observation and control preflight. Creation cannot be represented as that stopped-first sequence. This proposal makes the initial create-and-stop phase explicit while retaining the same Qwen experiment.

| Proposed allocation | Boundary |
|---|---|
| Provider / region | RunPod Secure Cloud / CA-MTL-1 only |
| GPU | Exactly 1 × NVIDIA A40, 48 GB |
| Storage | 80 GB container disk; 0 GB persistent volume; no network volume |
| Signed-in draft | $0.49/hour compute + $0.011/hour storage = $0.501/hour |
| Rate ceilings | Compute ≤ $0.49/hour; storage ≤ $0.012/hour |
| Entire operation | Cumulative ≤ $1.50; ≤ 60 minutes from first create request |
| Shutdown reserve | Both external timer and eventual pod watchdog trigger at the same outer deadline minus 300 seconds |

The signed-in draft was observed at `2026-09-24T17:19:54.070254+00:00`; it is historical evidence, not a reservation or a current price guarantee. Refresh capacity and actual rates before any future authorized allocation. At ceiling rates a full hour would cost $0.502 for this allocation; this estimate does not demonstrate successful shutdown or capacity. The separate existing 100 GB volume is excluded.

Proposed sequence: prepare and review the control procedure locally; obtain authorization for this exact provisioning boundary; arm an independent external shutdown controller before one create request; identify and immediately stop the newly created pod without loading a model; verify its stopped state and zero hourly charge; then perform the existing exact-pod stopped-state preflight. A later resume and inference stage requires reviewed integration and explicit authorization bound to the actual pod and execution plan. It must reuse the original outer deadline and remaining budget. Stopping the pod can release its GPU, so a later resume may fail even when creation succeeds. That failure ends this attempt; there is no replacement pod or second creation.

Read [PROTOCOL.md](PROTOCOL.md) and the machine-readable [proposal](configs/provisioning_proposal.json). The unresolved deadline/approval integration is a blocker, not an implemented safeguard. No scientific files, runner, resource proposal, prompts, parsers, reports, or split are changed here.

Bindings:

- Prepared plan: `c4586bb0c65163244dac80344ea96f691493a1b89a5911eefcba54f2384c74fc`
- Reviewed alternate execution plan: `ab82f51cf3883e93b2c33bf750a32148297226ebd3e9ada0febb54d18d85a4f5`
- Prior execution plan: `0223e5be543e0bb8f4493302476ade6cf53b5ee08653acf42858bd8263e249c5`
- Alternate resource proposal: `884dc38d2004abb44c64d2363e525969bf7f7fabd35f86eb5d03144bf34e72e2`

Provider lifecycle reference: [RunPod create-pod API](https://docs.runpod.io/api-reference/pods/POST/pods). The recorded local readiness check supplied the draft above; no provider calls were made to write this proposal.
