# Qwen live execution runtime: alternate GPU v1

## Read-only capacity watch

Use `scripts/capacity_watch.py` to check the five reviewed Secure Cloud GPU types and one exact stopped pod. The script sends a GraphQL query only and cannot provision or change resources. It deliberately reports `BLOCKED_CAPACITY_UNVERIFIED`, because regional stock and pod configuration snapshots do not prove exact-pod resumability. The command and checklist are in [OPERATOR_CHECKLIST.md](OPERATOR_CHECKLIST.md).

The HTTP client supplies an explicit application User-Agent, following the project's [previous control-client fix](../report_labeling_llm_v3_execution_v1/CONTROL_PREFLIGHT.md). Missing regional stock fields are reported as unknown, not as confirmed lack of capacity. The exact host's `machine_gpu_available` is reported separately. A zero value is a concrete reason to wait; a positive snapshot is not a reservation or guarantee of a successful resume.

This is a separately versioned, disabled-by-default runner for the previously reviewed five-report Qwen smoke. It reuses the frozen Qwen runtime, parsing, prepared token artifacts, process-group supervisor, watchdog, and the reviewed alternate-GPU gate. It adds no model or scientific condition.

The runner binds prepared-plan SHA-256 `c4586bb0c65163244dac80344ea96f691493a1b89a5911eefcba54f2384c74fc`, prior execution-plan SHA-256 `0223e5be543e0bb8f4493302476ade6cf53b5ee08653acf42858bd8263e249c5`, and alternate resource proposal SHA-256 `884dc38d2004abb44c64d2363e525969bf7f7fabd35f86eb5d03144bf34e72e2`. The new execution-plan file additionally pins this runtime policy and source, both resource gates, and the watchdog source.

The approved GPU allowlist is NVIDIA A40, NVIDIA RTX A6000, NVIDIA L40, NVIDIA RTX 6000 Ada Generation, and NVIDIA L40S. Each future grant selects exactly one named GPU, exact pod, and exact region. The signed-in rates must remain within that GPU's proposal ceiling and the storage ceiling. A different allowed GPU still fails. There is no fallback to another GPU, region, pod, or host.

The provider-observation receipt must match the grant, record `provider_state: "STOPPED"`, and be timestamped after approval and no more than 10 minutes before the requested start. It records pre-start state only; the stop preflight and live watchdog are the current control checks after start.

Real execution remains disabled in `configs/runtime.json`; this package creates no approval record and makes no provider calls. Do not change the execution switch as part of this source milestone.

See [PROTOCOL.md](PROTOCOL.md), [OPERATOR_CHECKLIST.md](OPERATOR_CHECKLIST.md), and [RESULTS.md](RESULTS.md).
