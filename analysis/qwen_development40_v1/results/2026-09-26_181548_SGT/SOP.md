# Reproducible live SOP — development benchmark of baseline A

This records the commands and corrections used successfully on 26 September
2026. Replace every uppercase placeholder with current, independently observed
values. Never reuse the pod identity, private receipts, key, rates or deadlines
from this run.

## 1. Freeze scope and start the original clock

Use the reviewed commit and verify both frozen hashes:

```sh
git checkout EXACT_REVIEWED_COMMIT
git rev-parse HEAD
sha256sum analysis/qwen_development40_v1/configs/execution_plan.json
sha256sum state/runs/qwen-development40-v1-private/prepared-control-a/plan.json
```

Record the actual container start rather than estimating it from when inference
begins:

```sh
ps -o lstart= -p 1
date -u
```

Create the private `0600` session from the user's approval. `hard_deadline` is
at most T0 + 3 hours and `shutdown_at` leaves the required reserve. If an
approximate T0 was used initially, create a new immutable session and shutdown
receipt from the exact PID-1 start; do not edit a session already bound to a
watchdog.

## 2. Reach an already-running pod without restarting it

If an account SSH key was added after the pod started, it may not be injected
into the running container. The successful recovery was:

1. Open the exact pod's signed-in **Jupyter Notebook** link.
2. Start one Jupyter terminal.
3. Append only the operator's public key to `/root/.ssh/authorized_keys`, then
   set directory mode `0700` and file mode `0600`.
4. Connect using the exact TCP host/port displayed by that pod. Use a dedicated
   `known_hosts` file and `StrictHostKeyChecking=yes` after first trust capture.

Do not put a private SSH key, Jupyter token, host, port or pod ID in Git.

## 3. Temporary RunPod key — successful direct-to-pod fallback

First follow [the main temporary-key SOP](../../../../docs/RUNPOD_TEMPORARY_KEY_SOP.md):
unique timestamped name, **Restricted**, GraphQL **Read / Write**, AI API
**None**. This is account-level authority until disabled.

The loopback helper returned `Refused` in both the in-app browser and the Chrome
extension path during this session. Do not keep creating keys after a failed
intake. Disable every lost key before a replacement. The successful fallback,
available only because the exact pod was already running and Jupyter was
reachable, was:

1. Create one replacement key and press its RunPod **Copy** button.
2. Keep the secret inside the browser automation session. Browser-session
   clipboard and the macOS clipboard are not assumed identical; do not use
   `Meta+V` as a bridge.
3. In the pod terminal run a silent, owner-only intake:

   ```sh
   umask 077
   mkdir -p /root/.config/rsna-knee/temporary-control
   read -rs RPKEY
   printf '%s\n' "$RPKEY" > /root/.config/rsna-knee/temporary-control/SESSION.key
   unset RPKEY
   chmod 600 /root/.config/rsna-knee/temporary-control/SESSION.key
   ```

4. UI automation of an xterm can wrap typed clipboard data in bracketed-paste
   markers: exact prefix `ESC[200~` and suffix `ESC[201~`. In this run a valid
   50-byte key became 62 bytes. Before using the file, inspect only metadata and
   booleans. If, and only if, both exact markers surround a string matching
   `rpa_[A-Za-z0-9_-]{20,250}`, strip exactly those markers, rewrite with mode
   `0600`, and revalidate. Never print the value.
5. Clear the browser clipboard variable immediately. Keep the signed-in RunPod
   Console open as the independent outside-pod stop route.

This fallback gives the pod-local watchdog its key but does not create a local
off-pod API copy. Loss of pod access therefore leaves the signed-in console as
the independent stop route. Do not claim otherwise.

## 4. Transfer only the reviewed source and private prepared package

Create the remote destination before SCP; this run's first SCP failed because
the ignored `state/` hierarchy did not exist.

```sh
git archive --format=tar HEAD | ssh REMOTE 'mkdir -p /workspace/rsna-knee-medgemma && tar -xf - -C /workspace/rsna-knee-medgemma'
ssh REMOTE 'mkdir -p /workspace/rsna-knee-medgemma/state/runs/qwen-development40-v1-private/SESSION'
scp -r state/runs/qwen-development40-v1-private/prepared-control-a REMOTE:/workspace/rsna-knee-medgemma/state/runs/qwen-development40-v1-private/
scp PRIVATE_SESSION_JSON REMOTE:/workspace/rsna-knee-medgemma/state/runs/qwen-development40-v1-private/SESSION/session.json
```

Recompute the execution-plan and prepared-plan hashes on the pod.

## 5. Arm shutdown before setup or model loading

```sh
python3 analysis/qwen_development40_v1/scripts/stop_at.py --arm \
  --session PRIVATE_SESSION_JSON --key PRIVATE_RUNPOD_KEY \
  --pod-id EXACT_POD_ID --plan-sha256 EXACT_EXECUTION_PLAN_SHA \
  --receipt PRIVATE_RECEIPT_JSON --result PRIVATE_STOP_RESULT_JSON \
  --cancel-marker PRIVATE_OPERATOR_STOP_MARKER
```

Verify the worker PID and exact command. If a corrected session creates a new
watchdog, verify the new worker before terminating the obsolete earlier worker.
The runner rechecks this process before model loading and every generation.

## 6. Install the pinned environment and cache

The base image already supplied Torch 2.8.0+cu128. Reuse it through a separate
system-site-packages venv and install the remaining exact versions from
`analysis/qwen_execution_v1/configs/requirements.txt`. Verify versions, CUDA
12.8, one visible GPU, native BF16 and at least 36 GiB free VRAM.

Download only the pinned public model revision:

```sh
. /workspace/qwen-dev40-venv/bin/activate
export HF_HOME=/workspace/hf-cache
hf download Qwen/Qwen3-14B \
  --revision 40c069824f4251a91eefaf281ebe4c544efd3e18 \
  --cache-dir /workspace/hf-cache
```

When `HF_HOME` changes, a token stored under the default home is not discovered.
This run warned that the request was unauthenticated, but the public download
completed. Future runs should either intentionally download anonymously or put
the approved token at the active HF home; never print it. Run the frozen cache
audit. It must verify all eight weight shards and the pinned revision before
offline mode is enabled.

## 7. Dry run, fresh observation and real run

Run the zero-generation parity check first. Then create a fresh private
`RUNNING` observation from the signed-in console: exact pod, region, Secure
Cloud, GPU/count, 80 GB container disk, zero persistent/network volume and the
actual compute/storage/total rates. Unknown fields remain blockers.

```sh
export HF_HOME=/workspace/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python analysis/qwen_development40_v1/scripts/run_dev40.py --run \
  --prepared PRIVATE_PREPARED_DIR --cache /workspace/hf-cache \
  --output NEW_PRIVATE_OUTPUT_DIR --session PRIVATE_SESSION_JSON \
  --observation PRIVATE_RUNNING_OBSERVATION_JSON \
  --shutdown-receipt PRIVATE_RECEIPT_JSON \
  --plan-sha256 EXACT_EXECUTION_PLAN_SHA
```

Monitor only durable counts, process state and GPU telemetry. The Transformers
warning about `temperature`, `top_p` and `top_k` being ignored was non-fatal;
the frozen run remained greedy. Do not change settings during the run.

## 8. Copy, verify, stop and disable

After completion, archive on the pod, create SHA-256, copy both files, verify
the archive locally, extract it and verify every digest in
`session_result.json`. Then call the reviewed single exact-pod `stop_once`.
Do not retry an unknown stop outcome blindly.

Independently refresh the signed-in console. Completion requires **Not running**
for compute and storage and **$0.00/hour** total. Finally locate the exact
temporary-key name and select **Disable key → Yes**; verify the row says
**Disabled**. Keep a private closeout receipt with timestamps, archive hash,
console observation, key status and estimated-versus-invoiced cost distinction.

## Errors observed in this run

| Error/observation | Resolution |
|---|---|
| Loopback key helper returned `Refused` in two browser paths | Stopped retrying the helper; disabled each lost key; used the documented direct-to-pod fallback. |
| Browser Copy and `Meta+V` used different clipboards | Kept the key in the browser-session clipboard and sent it directly to silent terminal input. |
| xterm added 12 bracketed-paste bytes | Verified exact wrapper plus key regex, stripped only that wrapper, rewrote mode `0600`. |
| Account SSH key added after pod start was not active | Added the public key through the already-authorized Jupyter terminal, then used direct SSH. |
| First SCP could not canonicalize the private destination | Created the ignored remote `state/` directories before SCP. |
| Initial session used an approximate T0 | Read PID-1 start, created a new immutable exact-start session and receipt, verified its watchdog, then ended the obsolete worker. |
| Cache audit appeared silent for over 30 seconds | It was hashing 29.5 GB of weights; bounded polling completed successfully. |
| 18 report-level parser failure records | Preserved them unchanged; no repair/retry. They correspond to 38 affected condition rows. |

This SOP reproduces the operator path; it does not authorize a new pod, budget,
credential or inference session.
