# MedGemma v3 · gated GPU execution proposal

**Setup attempted after user approval; stopped before model loading because the provider shutdown preflight failed.** See [the setup outcome](SETUP_ATTEMPT.md). This revision makes the completed local runner executable only with a fresh, explicit private authorization and a live self-stop watchdog. The prior runtime remains hard-disabled and unchanged.

| Item | Proposal |
|---|---|
| Objective | Compare the frozen control and evidence-first prompts |
| Input | Same five original-language development reports; no MRI images or organizer answers in prompts |
| Output | Twelve four-state proposals plus evidence/confidence, raw responses/tokens and execution receipts |
| Design | Two arms × two repeats, ABBA order, maximum 20 generations |
| GPU | One RTX 6000 Ada 48 GB, existing stopped pod |
| Cost | $0.84/hour compute + approximately $0.011/hour temporary disk |
| Proposed limit | $2 experiment budget, at most 90 minutes including setup and transfer |
| Validation/training | None |

Read [PROTOCOL.md](PROTOCOL.md), [RESULTS.md](RESULTS.md), [source review](REVIEW.md), and the [resource proposal](configs/resource_proposal.json). No credentials, model weights or private data are published.

From the checkout root, prepare a **new** private runtime plan:

```bash
python analysis/report_labeling_llm_v3_execution_v1/scripts/runtime_v3_execution.py prepare \
  --candidate "$RSNA_STATE_DIR/runs/report-labeling-llm-v3-candidate-20260915" \
  --state "$RSNA_STATE_DIR" \
  --prepared "$RSNA_STATE_DIR/runs/report-labeling-llm-v3-execution-prepared"
```

Preparation does not approve spending. After explicit user approval, the operator records the exact plan, quote, rates, timestamps and pod in a private authorization file. Once the pod-local watchdog is armed and verified, the bounded command is:

```bash
python analysis/report_labeling_llm_v3_execution_v1/scripts/runtime_v3_execution.py run \
  --prepared "$RSNA_STATE_DIR/runs/report-labeling-llm-v3-execution-prepared" \
  --plan-sha256 REVIEWED_SHA --cache "$RSNA_STATE_DIR/cache" \
  --authorization "$RSNA_STATE_DIR/private-authorization.json" \
  --watchdog "$RSNA_STATE_DIR/private-watchdog.json" --execute
```

Both the parent and worker reject missing or invalid approval. Scripts do not provision a server, buy credits or download weights. Model generation uses the existing full-cache checksum audit and offline adapter. Every real session is bounded by the original 60-minute limit and the approved provider deadline with a copy/stop reserve.

The execution environment uses the [ten exact package pins](configs/requirements.txt), rather than the older MRI-pilot bootstrap recipe. On the approved CUDA host, install PyTorch 2.8.0 from the official CUDA 12.8 wheel index first, then these requirements. Verify all versions before downloading the pinned MedGemma snapshot. Arm the stdlib-only watchdog before lengthy setup.

For CPU-only software checks, `runtime_v3_execution.py demo --prepared NEW_PRIVATE_PATH` fabricates five reports and fixed JSON responses without loading a model. `review_v3_execution.py bundle` creates the same blinded workflow as the reviewed local runtime. Fabricated sessions require `--allow-synthetic-demo` and cannot be presented as measured outputs.
