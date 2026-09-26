# First Kaggle T4 diagnostic and local correction

Executed source: `ef2dcd3ce1fd0dfc74a1f34d9d4f55c3a2a7b58c`.
This is a failed pre-load diagnostic, not a model inference result.

The private dataset uploaded successfully. The offline Kaggle session verified
32 asset checksums and installed the wheelhouse in 16.5 seconds. The visible
allocation was two T4 GPUs with about 15 GiB each. The start request was at
20:05:28 SGT; the session was observed running during 20:11 SGT and confirmed
stopped by 20:14:55 SGT. Displayed Kaggle quota changed from 00:00 to 00:03 of
30 hours. These minute-resolution UI values are not second-resolution billing.

The environment cell raised:

```text
ContractError: Package version drift for huggingface-hub: expected 0.36.0, observed None
```

The builder's eight-name inventory omitted `huggingface-hub` and `tokenizers`,
although the validator requires all ten configured pins. Thus `None` was a
missing dictionary entry, not evidence of an incorrect installed version.
No model was loaded and no forward, numerical diagnostic, example prediction,
submission or leaderboard score was produced. Exact environment strings and
model memory measurements were not printed before the exception and remain
unavailable. Cross-precision parity is NUMERICAL_PARITY_UNVERIFIED.

The prospective local correction collects every configured pin and writes
`environment_observed.json`, including setup timings and free GPU memory,
before validation. It preserves all hardware/version validators, model code,
configuration, preprocessing, adapter and assets. A CPU regression executes the
generated inventory/validation statements with synthetic CUDA observations,
proves every pin is collected, and proves real version drift in either formerly
omitted package still fails with the observed environment preserved. All 19
focused tests pass; these tests are not GPU inference evidence.

The original notebook and outputs remain unchanged. Private cell-output capture,
session JSON and hashes are under `state/kaggle_image_t4_v1/runs/20260926T120528Z/`.
Private Kaggle Quick Save version 2 retains the failed cells without rerunning.
The corrected notebook uses a new filename and hash documented in the README.
No second GPU session has been started.
