"""Two independent, single-visible-T4 workers using byte-frozen Version 8 code."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd

from inference_core import (LABELS, ContractError, atomic_json, read_config,
                            sha256_file, validate_free_memory,
                            validate_submission, validate_tables)

V8_COMMIT = '4a4382ab8546cd872e90a3e81654291be9fa7546'
V8_HASHES = {
    'inference_core.py': '2b6189ec846bad84c79642697dbcbb86101d31bfcd19857f3fec350e3c98ab0e',
    'submission_runtime.py': 'bc0d4bd51e72637adbe8fedf51a860e3e88c96d976800f1bdb73c3b8a7c2c6a8',
    'numerical_runtime.py': 'c9fbbf1f1c781070431c732e040dd8f961dfb851550fbac42f5f445e3c408844',
    'inference_config.json': '0b57546facec15fb5c2c476faa60e3c4e855013d38e0d4bd8805f9bdbd6ed742',
}
PARITY_ATOL = 1e-6  # Fixed before the diagnostic; relative tolerance is zero.


def verify_v8_sources(config_path):
    root = Path(config_path).resolve().parent
    for name, expected in V8_HASHES.items():
        if sha256_file(root / name) != expected:
            raise ContractError(f'Frozen Version 8 source drift: {name}')


def deterministic_shards(test):
    ids = test['StudyInstanceUID']
    if len(ids) < 2 or ids.isna().any() or not ids.is_unique:
        raise ContractError('Two replicas require at least two unique, non-null studies')
    ordered = sorted(ids.astype(str).tolist())
    return [ordered[0::2], ordered[1::2]]


def compare_reference(prediction, test, reference_path, reference_sha256, atol=PARITY_ATOL):
    if not math.isfinite(atol) or atol < 0 or atol > PARITY_ATOL:
        raise ContractError('Cannot relax the predeclared parity tolerance')
    if sha256_file(Path(reference_path)) != reference_sha256:
        raise ContractError('Version 8 reference checksum mismatch')
    reference = pd.read_csv(reference_path, dtype={'StudyInstanceUID': str})
    validate_submission(reference, test)
    validate_submission(prediction, test)
    order = test.StudyInstanceUID.tolist()
    actual = prediction.set_index('StudyInstanceUID').loc[order, LABELS].to_numpy(float)
    expected = reference.set_index('StudyInstanceUID').loc[order, LABELS].to_numpy(float)
    difference = np.abs(actual - expected)
    result = {'reference_sha256': reference_sha256, 'atol': atol, 'rtol': 0,
              'compared_scores': int(difference.size),
              'max_absolute_difference': float(difference.max()),
              'exact_scores': int((difference == 0).sum()),
              'passed': bool((difference <= atol).all())}
    if not result['passed']:
        raise ContractError('Version 8 score parity failed: ' + json.dumps(result))
    return result


def _worker_command(request_path, worker_id, worker_dir):
    return [sys.executable, '-u', str(Path(__file__).resolve()), '--worker',
            str(request_path), '--worker-id', str(worker_id), '--worker-dir', str(worker_dir)]


def _kill_and_reap(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=15)


def _sample_gpu_utilization(elapsed):
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=2)
    if result.returncode:
        raise RuntimeError('nvidia-smi utilization sampling failed')
    return [{'elapsed_seconds': elapsed, 'gpu_index': int(row[0]),
             'gpu_utilization_percent': float(row[1]), 'memory_used_mib': float(row[2])}
            for row in csv.reader(result.stdout.splitlines())]


def _validated_receipt(row, worker_dir, shard, test):
    try:
        result = json.loads((worker_dir / 'worker_result.json').read_text())
        if not isinstance(result, dict):
            raise ValueError('Expected an object receipt')
    except (OSError, ValueError) as error:
        raise ContractError('Worker result missing/invalid; no retry') from error
    for key, expected in {'worker_id': row['worker_id'], 'pid': row['pid'],
                          'visible_devices': str(row['worker_id']), 'gpu_count': 1,
                          'placement': 'single_gpu', 'status': 'complete',
                          'phase': 'inference', 'study_ids': shard}.items():
        if result.get(key) != expected:
            raise ContractError(f'Worker receipt mismatch: {key}')
    path = worker_dir / 'example' / 'submission.csv'
    if Path(result['submission_path']).resolve() != path.resolve():
        raise ContractError('Worker submission path mismatch')
    if sha256_file(path) != result['submission_sha256']:
        raise ContractError('Worker submission checksum mismatch')
    prediction = pd.read_csv(path, dtype={'StudyInstanceUID': str})
    shard_test = test.set_index('StudyInstanceUID', drop=False).loc[shard].reset_index(drop=True)
    validate_submission(prediction, shard_test)
    return result, prediction


def run_replicas(request_path, deadline_monotonic, cuda=None):
    """Launch exactly two children, fail as a group, merge only complete results."""
    request_path = Path(request_path).resolve()
    request = json.loads(request_path.read_text())
    verify_v8_sources(request['config_path'])
    config = read_config(Path(request['config_path']))
    test, _, _ = validate_tables(Path(request['data_dir']))
    shards = deterministic_shards(test)
    # Reference is only used by the parent for post-inference parity, never a model input.
    if sha256_file(Path(request['reference_csv'])) != request['reference_sha256']:
        raise ContractError('Version 8 reference checksum mismatch before launch')
    validate_submission(pd.read_csv(request['reference_csv'], dtype={'StudyInstanceUID': str}), test)
    sample_utilization = cuda is None
    if cuda is None:
        import torch
        cuda = torch.cuda
    if cuda.device_count() != 2:
        raise ContractError('Exactly two visible T4s are required for independent replicas')
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, '', '0,1'):
        raise ContractError('Ambiguous parent CUDA device mapping; refusing remapping')
    for i in range(2):
        cuda.synchronize(i)
    free = [cuda.mem_get_info(i)[0] / 2**30 for i in range(2)]
    validate_free_memory(free, config['minimum_free_gpu_gib_before_load'])
    if deadline_monotonic <= time.monotonic():
        raise ContractError('Shared session deadline already expired')
    root = Path(request['output_root']).resolve()
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = {'status': 'running', 'shard_policy': 'sorted_uid_alternating_0_1',
              'free_vram_before_gib': free, 'workers': [], 'scientific_source_commit': V8_COMMIT}
    processes, logs = [], []
    samples, sample_errors = [], []
    last_sample = float('-inf')

    def persist():
        atomic_json(root / 'replica_status.json', status)

    try:
        for i in range(2):
            directory = root / f'worker_{i}'
            directory.mkdir(exist_ok=False)
            log = (directory / 'worker.log').open('x')
            logs.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
            process = subprocess.Popen(_worker_command(request_path, i, directory), env=env,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(process)
            status['workers'].append({'worker_id': i, 'pid': process.pid,
                                      'physical_gpu_index': i, 'study_ids': shards[i],
                                      'started_after_parent_seconds': time.monotonic()-started,
                                      'status': 'running'})
            persist()
            print(f'REPLICA_STARTED worker={i} studies={len(shards[i])} pid={process.pid}', flush=True)
        pending = {0, 1}
        while pending:
            if time.monotonic() >= deadline_monotonic:
                raise ContractError('Hard session deadline: killing both replica process groups')
            for i in sorted(pending.copy()):
                returncode = processes[i].poll()
                if returncode is None:
                    continue
                processes[i].wait()
                row = status['workers'][i]
                row.update(returncode=returncode, process_reaped=True,
                           exited_after_parent_seconds=time.monotonic()-started)
                pending.remove(i)
                persist()
                if returncode != 0:
                    raise ContractError(f'Replica {i} failed (exit {returncode}); see worker_{i}/worker.log')
                result, _ = _validated_receipt(row, root/f'worker_{i}', shards[i], test)
                row.update(status='complete', result=result)
                persist()
                print(f'REPLICA_COMPLETE worker={i} studies={len(shards[i])}', flush=True)
            if (sample_utilization and time.monotonic()-last_sample >= 2
                    and deadline_monotonic-time.monotonic() > 2.5):
                last_sample = time.monotonic()
                try:
                    samples.extend(_sample_gpu_utilization(last_sample-started))
                except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
                    sample_errors.append({'elapsed_seconds':last_sample-started, 'error':str(error)})
                atomic_json(root/'gpu_utilization.json', {'samples':samples,'errors':sample_errors})
            if pending:
                time.sleep(min(0.2, max(0, deadline_monotonic-time.monotonic())))
        pieces = []
        for i, row in enumerate(status['workers']):
            _, prediction = _validated_receipt(row, root/f'worker_{i}', shards[i], test)
            pieces.append(prediction)
        combined = pd.concat(pieces, ignore_index=True)
        validate_submission(combined, test)
        combined = combined.set_index('StudyInstanceUID').loc[test.StudyInstanceUID].reset_index()
        schema = validate_submission(combined, test)
        # Keep outputs on failure, but never promote partial or non-parity predictions.
        combined.to_csv(root/'combined_candidate.csv', index=False)
        parity = compare_reference(combined, test, Path(request['reference_csv']), request['reference_sha256'])
        reference = pd.read_csv(request['reference_csv'], dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
        diffs = combined.copy()
        for label in LABELS:
            diffs[label] = combined[label].to_numpy()-reference.loc[combined.StudyInstanceUID, label].to_numpy()
        diffs.to_csv(root/'parity_differences.csv', index=False)
        atomic_json(root/'parity.json', parity)
        if time.monotonic() >= deadline_monotonic:
            raise ContractError('Deadline reached before final submission publication')
        final = root/'submission.csv'
        # Hard-link a complete candidate, atomically, without replacing any prior file.
        os.link(root/'combined_candidate.csv', final)
        elapsed = time.monotonic()-started
        status.update(status='THROUGHPUT_READY', final_submission_path=str(final),
                      submission_sha256=sha256_file(final), schema=schema, parity=parity,
                      parallel_wall_seconds=elapsed,
                      seconds_per_completed_study_including_worker_setup=elapsed/len(test))
        for row in status['workers']:
            life = row['exited_after_parent_seconds']-row['started_after_parent_seconds']
            row['process_lifetime_fraction_of_parallel_wall'] = life/elapsed
            row['full_study_work_fraction_of_parallel_wall'] = row['result'].get('example_status',{}).get('total_seconds',0)/elapsed
        persist()
        return status
    except BaseException as error:
        for process in processes:
            _kill_and_reap(process)
        for row, process in zip(status['workers'], processes):
            row.update(returncode=process.returncode, process_reaped=True)
        status.update(status='failed', error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc(), parallel_wall_seconds=time.monotonic()-started)
        persist()
        raise
    finally:
        for log in logs:
            log.close()


def worker(request_path, worker_id, worker_dir):
    """Import Torch only after this interpreter has one assigned visible GPU."""
    directory = Path(worker_dir)
    request = json.loads(Path(request_path).read_text())
    receipt = {'worker_id': worker_id, 'pid': os.getpid(), 'phase': 'prepare',
               'status': 'running', 'placement': 'single_gpu',
               'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
    receipt_path = directory/'worker_result.json'
    atomic_json(receipt_path, receipt)
    try:
        if worker_id not in (0, 1) or receipt['visible_devices'] != str(worker_id):
            raise ContractError('Worker CUDA visibility must exactly match its replica index')
        verify_v8_sources(request['config_path'])
        import torch
        from inference_core import validate_hardware
        from submission_runtime import prepare_t4_teacher, run_inference
        receipt['gpu_count'] = torch.cuda.device_count()
        if receipt['gpu_count'] != 1:
            raise ContractError('A replica must see exactly one GPU')
        config = read_config(Path(request['config_path']))
        prop = torch.cuda.get_device_properties(0)
        device = {'name': torch.cuda.get_device_name(0),
                  'capability_major': prop.major, 'capability_minor': prop.minor,
                  'total_gib': prop.total_memory/2**30}
        validate_hardware([device], config)
        receipt['visible_gpu'] = device
        test, series, sample = validate_tables(Path(request['data_dir']))
        shard = deterministic_shards(test)[worker_id]
        receipt['study_ids'] = shard
        shard_test = test.set_index('StudyInstanceUID', drop=False).loc[shard].reset_index(drop=True)
        base, adapter, data = Path(request['base_dir']), Path(request['adapter_dir']), Path(request['data_dir'])
        teacher, load_seconds, diagnostic, attempts = prepare_t4_teacher(
            config, series, data, base, adapter, str(test.StudyInstanceUID.iloc[0]),
            directory/'model_attempt.json', only_placement='single_gpu')
        atomic_json(directory/'frozen_runtime.json', {'load_seconds':load_seconds,
                    'diagnostic':diagnostic, 'attempts':attempts,
                    'hf_device_map':teacher.device_map, 'dtype_report':teacher.dtype_report})
        receipt.update(phase='inference', load_seconds=load_seconds)
        atomic_json(receipt_path, receipt)
        _, result = run_inference(config, shard_test, series, sample, data, base, adapter,
            directory/'example', teacher, load_seconds, request['asset_verification_seconds'],
            request['dependency_install_seconds'], diagnostic)
        path = directory/'example'/'submission.csv'
        receipt.update(status='complete', example_status=result, submission_path=str(path),
                       submission_sha256=sha256_file(path))
        atomic_json(receipt_path, receipt)
        return 0
    except BaseException as error:
        receipt.update(status='failed', error_type=type(error).__name__, error=str(error),
                       traceback=traceback.format_exc())
        atomic_json(receipt_path, receipt)
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=Path, required=True)
    parser.add_argument('--worker-id', type=int, choices=(0, 1), required=True)
    parser.add_argument('--worker-dir', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(worker(args.worker, args.worker_id, args.worker_dir))
