# Control credential preflight · 15 September 2026

**Local read and stop checks passed with the GPU already stopped.** The user explicitly approved a temporary RunPod control key with GraphQL Read/Write and no AI API access, including local testing, use for the experiment's shutdown timer and subsequent revocation. The console warned that GraphQL write permission covers account resources rather than only one Pod. No credential value is included in this repository or the receipts.

The first local request with Python's default HTTP client identification returned HTTP 403. With an explicit `User-Agent: rsna-preflight/1.0`, the same local checker and credential successfully queried the exact stopped Pod and received an EXITED result from the subsequent stop request. The [helper](../../scripts/runpod_stopped_preflight.py) now supplies this header; a seventh unit test checks it. This observed difference does not establish the provider-side cause of every earlier 403 or prove that all pod-scoped credentials are incapable of self-stop.

The live result confirmed the target identity, read access, acceptance of stop on the already-stopped target, and EXITED state. It started no compute. It does not guarantee a later running Pod will stop, and it does not replace the remote watchdog checks or external stop verification.

A subsequent attempt to resume the same approved Pod was unsuccessful. The documented GraphQL resume operation returned: “There are not enough free GPUs on the host machine to start this pod.” The Pod remained EXITED. This was a capacity failure after the local authentication issue had been resolved, not another authentication failure or a model result.

The original inference plan, prompt/split fingerprints and execution guards remain unchanged. No model weights were downloaded and no inference ran. Further allocation flexibility within the existing total $2 budget and original 90-minute window has been requested; no new replacement was provisioned during this recovery step. Approximately $0.10 was spent in the earlier setup; no additional compute started here.

## Later outcome

The user then explicitly approved replacement-host flexibility within the original total budget/deadline. A compatible replacement ran the unchanged plan with a verified live watchdog. Its final [measured outcome](RESULTS.md) is incomplete because of candidate truncation. The provider was stopped and the temporary key disabled afterward. This historical stopped-pod preflight record is not evidence that the watchdog CLI stop itself fired.
