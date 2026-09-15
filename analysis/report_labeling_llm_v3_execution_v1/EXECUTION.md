# Approved execution record · 15 September 2026

The user approved the original $2 total / 90-minute experiment and subsequently approved replacement RTX 6000 Ada hosts within the **same total budget and original 09:30 UTC (17:30 Singapore) deadline**. The replacement flexibility and temporary account control credential were explicit user approvals. These later operational authorizations superseded the initial proposal's single-host restriction; the frozen protocol, generation recipe and reviewed plan were not edited.

The earlier setup failure remains documented separately in [SETUP_ATTEMPT.md](SETUP_ATTEMPT.md). Its approximately $0.10 estimated cost counts toward the total. Capacity failures during subsequent allocation attempts returned no allocation; they were not model runs. A Community offer with insufficient CUDA support was not deployed. The successful replacement used Secure Cloud with the same RTX 6000 Ada and rate ceiling.

## Actual environment

| Item | Observed |
|---|---|
| GPU | NVIDIA RTX 6000 Ada Generation, 49,140 MiB |
| Allocated CPU / RAM | 16 vCPU (AMD EPYC 75F3 host) / 62 GB, console allocation |
| Driver / CUDA | 570.124.06 / 12.8 |
| Container | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` |
| Python | 3.12.3 |
| PyTorch | 2.8.0+cu128 |
| Inference | BF16, native support verified; SDPA; batch one; no adapter/quantization |
| Package pins | All ten [required versions](configs/requirements.txt) verified in runtime receipts |
| Storage | 80 GB temporary container disk; zero persistent volume |
| Compute / temporary disk | $0.84/hour + approximately $0.011/hour |
| Container start request | 08:41:28 UTC |
| Watchdog armed | 08:43:05 UTC |
| Parent dispatch | 08:50:26 UTC |
| Deadline / generation reserve | 09:30 UTC / last five minutes reserved for transfer and stop |

The image's system Python rejected direct package installation under PEP 668. An isolated virtual environment with system-site-packages reused the verified PyTorch installation; the remaining exact pins installed there. Both failed installation logs and the successful environment log are retained privately. No dependency pin was relaxed. Only the 13-file pinned MedGemma snapshot was downloaded, about 8.60 GB of weights plus assets; no MRI or Qwen data was fetched.

## Provenance and controls

The remote source archive is commit `9c311be8312e37aa5e2ed5545fb5c9ee708468e8`. Its execution scripts/configs/tests/protocol match the reviewed source `43d2a6396b2e7e691473f6b52c70716809d190ce`. Later public changes added operational documentation and an external stopped-pod preflight helper, not changes to the running experiment.

Runtime plan SHA-256: `6bcbee08e0bdb30f148f6a7a84c75afdcd327dc45c94b03b82724f0b625f1ee1`.
Resource proposal SHA-256: `6905a3ea42345bcaa20767e2990595fe22a07fa7e79b2ce0853a5c7a451f0f01`.
Model: `google/medgemma-1.5-4b-it` at `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b`.

Exact [control prompt](../report_labeling_llm_v3/prompts/control_v2.txt), [evidence-first prompt](../report_labeling_llm_v3/prompts/evidence_first.txt), [definitions](../report_labeling_llm_v3/prompts/target_definitions.txt) and [decoding design](../report_labeling_llm_v3/PROTOCOL.md) were preserved. Effective generation settings and tokenizer/chat-template checksums are recorded for each started run. All required cache bytes were audited before each model load; inference was offline. Raw responses were fsynced before parsing. No prompt repairs, retries or preferred-repeat selection were performed.

The approved temporary account key passed GraphQL read and stop against an already-stopped pod. The live watchdog separately verified current CLI syntax and CLI read access, then remained armed on the matching pod until shutdown. A stopped-pod GraphQL stop test does **not** establish that the watchdog's CLI stop path fired successfully. The key was supplied to the watchdog process environment; it was never written into the checkout, receipt, public source or model inputs. Provider identifiers, SSH details, grant/receipt hashes and exact process identity remain in the private provenance package.

The executed configuration preserves inherited descriptive fields such as `UNEXECUTED_RUNTIME` inside its frozen model dictionary. Actual execution is established by `synthetic=false` session records, verified GPU/backend metadata and durable model responses, not those static template labels.

See [RESULTS.md](RESULTS.md) for final measured outcomes and lifecycle completion, and [diagnostics](diagnostics/README.md) for the local input/output dashboard. No clinical correctness, semantic superiority or scaling permission follows from successful execution or technical acceptance.
