"""Local replay of saved V10 predictions through the real submission merge.

No model is loaded. This proves output assembly parity, not a new GPU run.
Private input IDs/results never become notebook dependencies.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FROZEN_FINAL_SHA = '8b66aef38061782800d830dd1c4bce4e21024a0ea34c6e404c98b9bafdb1a7ec'


def replay(v10_dir: Path, output_dir: Path) -> dict:
    sys.path.insert(0, str(REPO / 'analysis/kaggle_image_t4_v1/scripts'))
    spec = importlib.util.spec_from_file_location('submission_replay_runner', HERE / 'submission_runner.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    v10_dir, output_dir = Path(v10_dir), Path(output_dir)
    original = v10_dir / 'submission.csv'
    if hashlib.sha256(original.read_bytes()).hexdigest() != FROZEN_FINAL_SHA:
        raise ValueError('Expected unchanged downloaded V10 final CSV')
    baseline = pd.read_csv(original, dtype={'StudyInstanceUID': str})
    test = baseline[['StudyInstanceUID']].copy()
    runner.validate_submission(baseline, test)
    pieces, scored = [], []
    for i, shard in enumerate(runner.deterministic_shards(test)):
        root = v10_dir / f'worker_{i}'
        receipt = json.loads((root / 'worker_result.json').read_text())
        path = root / 'example/submission.csv'
        if runner.sha256_file(path) != receipt['submission_sha256']:
            raise ValueError('Saved worker CSV checksum changed')
        if receipt['study_ids'] != shard or receipt['status'] != 'complete':
            raise ValueError('Saved V10 shard mismatch')
        piece = pd.read_csv(path, dtype={'StudyInstanceUID': str})
        expected = test.set_index('StudyInstanceUID', drop=False).loc[shard].reset_index(drop=True)
        runner.validate_submission(piece, expected)
        records = json.loads((root / 'score_logits.json').read_text())['scores']
        expected_keys = [(uid, target) for uid in shard for target in runner.LABELS]
        if [(r['study_id'], r['label']) for r in records] != expected_keys:
            raise ValueError('Saved score-record identity/target order mismatch')
        for row in records:
            if not all(math.isfinite(float(row[k])) for k in
                       ('native_no_logit', 'native_yes_logit', 'yes_probability')):
                raise ValueError('Saved nonfinite score/logit')
        pieces.append(piece)
        scored.extend(records)
    combined = runner.merge_predictions(pieces, test)
    data = combined.to_csv(index=False).encode()
    # Compare actual CSV bytes, avoiding different CSV reader rounding policies.
    if data != original.read_bytes():
        raise ValueError('Submission merge bytes differ from downloaded V10')
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / 'replayed_submission.csv').write_bytes(data)
    result = {'status': 'V10_SAVED_OUTPUT_REPLAY_PASS', 'mode': 'LOCAL_REPLAY_NOT_NEW_INFERENCE',
              'studies': len(test), 'scores': len(scored),
              'native_logits_finite': True, 'byte_identical_final_csv': True,
              'maximum_score_difference': 0.0, 'submission_sha256': FROZEN_FINAL_SHA,
              'model_loaded': False, 'gpu_used': False,
              'runtime_reference_dependency': False}
    (output_dir / 'replay_result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--v10-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.v10_dir, args.output_dir), indent=2))
