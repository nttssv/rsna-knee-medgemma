"""Submission-only orchestration of the frozen V10 cached image inference path.

The frozen ImageTeacher, numerical runtime, configuration, and same-study cache
are loaded unchanged. Preparation does not score an example or run the repeated
diagnostic: every model forward belongs to one of the actual test targets.
"""
from __future__ import annotations

import argparse
import csv
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
                            validate_hardware, validate_submission, validate_tables)

V8_COMMIT = '4a4382ab8546cd872e90a3e81654291be9fa7546'
V10_COMMIT = '26dafd65b91e054577110f4c695f7d72a7660faf'
V8_HASHES = {
    'inference_core.py': '2b6189ec846bad84c79642697dbcbb86101d31bfcd19857f3fec350e3c98ab0e',
    'submission_runtime.py': 'bc0d4bd51e72637adbe8fedf51a860e3e88c96d976800f1bdb73c3b8a7c2c6a8',
    'numerical_runtime.py': 'c9fbbf1f1c781070431c732e040dd8f961dfb851550fbac42f5f445e3c408844',
    'inference_config.json': '0b57546facec15fb5c2c476faa60e3c4e855013d38e0d4bd8805f9bdbd6ed742',
}
CACHE_SHA256 = 'b15b4bd8608e32ef85ef709f1dc8a5c19be71db95ae088e45236dd2b2afc3980'
CACHE_POLICY = 'same_study_exact_pixel_vision_cache_v1'


def verify_cache_source(request):
    path = Path(request['config_path']).resolve().parent / 'vision_cache.py'
    if (sha256_file(path) != CACHE_SHA256
            or request.get('vision_cache_sha256', CACHE_SHA256) != CACHE_SHA256):
        raise ContractError('Frozen Version 10 vision cache source checksum mismatch')


def verify_v8_sources(config_path):
    root = Path(config_path).resolve().parent
    for name, expected in V8_HASHES.items():
        if sha256_file(root / name) != expected:
            raise ContractError(f'Frozen Version 8 source drift: {name}')


def deterministic_shards(test):
    if list(test.columns) != ['StudyInstanceUID']:
        raise ContractError('test.csv must contain only StudyInstanceUID')
    ids = test['StudyInstanceUID']
    if (len(ids) == 0 or ids.isna().any() or not ids.is_unique
            or any(not isinstance(uid, str) or not uid.strip() for uid in ids)):
        raise ContractError('Test studies must be nonempty, unique, non-null string IDs')
    ordered = sorted(ids.tolist())
    return [ordered[0::2], ordered[1::2]]


def merge_predictions(pieces, test):
    """Validate complete coverage and preserve actual test.csv row order."""
    deterministic_shards(test)
    combined = pd.concat(pieces, ignore_index=True)
    validate_submission(combined, test)
    combined = combined.set_index('StudyInstanceUID').loc[test.StudyInstanceUID].reset_index()
    validate_submission(combined, test)
    return combined


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


def _validate_cache(cache, shard):
    for key, expected in {'policy': CACHE_POLICY, 'misses': len(shard),
                          'hits': 11 * len(shard), 'underlying_vision_calls': len(shard),
                          'cache_released': True, 'active_study': None,
                          'decoder_forwards_per_study': 12}.items():
        if cache.get(key) != expected:
            raise ContractError(f'Worker vision cache accounting mismatch: {key}')
    studies = cache.get('studies')
    if not isinstance(studies, list) or len(studies) != len(shard):
        raise ContractError('Worker vision cache study coverage mismatch')
    for row, uid in zip(studies, shard):
        if not isinstance(row, dict) or any(row.get(key) != expected for key, expected in
                {'study_id': uid, 'misses': 1, 'hits': 11}.items()):
            raise ContractError('Worker vision cache per-study accounting mismatch')


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
                          'phase': 'inference', 'study_ids': shard,
                          'score_record_count': 12 * len(shard),
                          'model_loaded': bool(shard)}.items():
        if result.get(key) != expected:
            raise ContractError(f'Worker receipt mismatch: {key}')
    _validate_cache(result.get('vision_cache', {}), shard)
    inference = result.get('inference_status', {})
    if (inference.get('status') != 'complete'
            or inference.get('completed_studies') != len(shard)):
        raise ContractError('Worker inference status is incomplete')
    path = worker_dir / 'inference' / 'submission.csv'
    if Path(result['submission_path']).resolve() != path.resolve():
        raise ContractError('Worker submission path mismatch')
    if sha256_file(path) != result['submission_sha256']:
        raise ContractError('Worker submission checksum mismatch')
    prediction = pd.read_csv(path, dtype={'StudyInstanceUID': str})
    shard_test = test.set_index('StudyInstanceUID', drop=False).loc[shard].reset_index(drop=True)
    if shard:
        validate_submission(prediction, shard_test)
    elif len(prediction) or list(prediction.columns) != ['StudyInstanceUID', *LABELS]:
        # Frozen validate_submission reduces min/max and therefore rejects an
        # empty array. Only an explicitly empty assigned shard takes this path.
        raise ContractError('Empty worker must produce a header-only submission')
    if prediction.StudyInstanceUID.tolist() != shard:
        raise ContractError('Worker submission study order differs from its shard')
    return result, prediction


def _study_times(worker_dir, shard):
    timing = pd.read_csv(worker_dir / 'inference' / 'study_timings.csv',
                         dtype={'StudyInstanceUID': str})
    if timing['StudyInstanceUID'].tolist() != shard:
        raise ContractError('Complete-study timing coverage/order mismatch')
    values = timing['study_seconds'].to_numpy(float)
    if len(values) != len(shard) or not np.isfinite(values).all() or (values <= 0).any():
        raise ContractError('Invalid complete-study timing evidence')
    return values.tolist()


def run_replicas(request_path, deadline_monotonic, cuda=None):
    """Launch exactly two children, fail as a group, merge only complete results."""
    request_path = Path(request_path).resolve()
    request = json.loads(request_path.read_text())
    verify_v8_sources(request['config_path'])
    verify_cache_source(request)
    config = read_config(Path(request['config_path']))
    test, _, _ = validate_tables(Path(request['data_dir']))
    shards = deterministic_shards(test)
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
    if not all(math.isfinite(value) for value in free):
        raise ContractError('GPU free memory evidence must be finite')
    validate_free_memory(free, config['minimum_free_gpu_gib_before_load'])
    if not math.isfinite(deadline_monotonic) or deadline_monotonic <= time.monotonic():
        raise ContractError('Shared session deadline already expired or invalid')
    root = Path(request['output_root']).resolve()
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = {'status': 'running', 'shard_policy': 'sorted_uid_alternating_0_1',
              'free_vram_before_gib': free, 'workers': [],
              'planned_studies': len(test), 'scientific_source_commit': V8_COMMIT,
              'vision_cache_source_commit': V10_COMMIT,
              'vision_cache_sha256': CACHE_SHA256}
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
                                      'started_after_parent_seconds': time.monotonic() - started,
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
                           exited_after_parent_seconds=time.monotonic() - started)
                pending.remove(i)
                persist()
                if returncode != 0:
                    raise ContractError(f'Replica {i} failed (exit {returncode}); see worker_{i}/worker.log')
                result, _ = _validated_receipt(row, root / f'worker_{i}', shards[i], test)
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
                time.sleep(min(0.2, max(0, deadline_monotonic - time.monotonic())))
        pieces, study_times = [], []
        for i, row in enumerate(status['workers']):
            _, prediction = _validated_receipt(row, root / f'worker_{i}', shards[i], test)
            pieces.append(prediction)
            study_times.extend(_study_times(root / f'worker_{i}', shards[i]))
        combined = merge_predictions(pieces, test)
        schema = validate_submission(combined, test)
        combined.to_csv(root / 'combined_candidate.csv', index=False, mode='x')
        if time.monotonic() >= deadline_monotonic:
            raise ContractError('Deadline reached before final submission publication')
        final = root / 'submission.csv'
        # Publish a complete candidate atomically without replacing an existing file.
        os.link(root / 'combined_candidate.csv', final)
        elapsed = time.monotonic() - started
        status.update(status='complete', final_submission_path=str(final),
                      submission_sha256=sha256_file(final), schema=schema,
                      completed_studies=len(test), complete_study_seconds=study_times,
                      maximum_complete_study_seconds=max(study_times),
                      parallel_wall_seconds=elapsed,
                      seconds_per_completed_study_including_worker_setup=elapsed / len(test))
        for row in status['workers']:
            life = row['exited_after_parent_seconds']-row['started_after_parent_seconds']
            row['process_lifetime_fraction_of_parallel_wall'] = life/elapsed
            row['full_study_work_fraction_of_parallel_wall'] = row['result'].get('inference_status',{}).get('total_seconds',0)/elapsed
        persist()
        return status
    except BaseException as error:
        for process in processes:
            _kill_and_reap(process)
        for row, process in zip(status['workers'], processes):
            row.update(returncode=process.returncode, process_reaped=True)
        status.update(status='failed', error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc(), parallel_wall_seconds=time.monotonic() - started)
        persist()
        raise
    finally:
        for log in logs:
            log.close()


def _free_memory_guard(torch, minimum):
    torch.cuda.synchronize(0)
    free = [torch.cuda.mem_get_info(0)[0] / 2**30]
    if not all(math.isfinite(value) for value in free):
        raise ContractError('GPU free memory evidence must be finite')
    validate_free_memory(free, minimum)
    return free


def load_submission_teacher(config, series, data_dir, base_dir, adapter_dir,
                            attempt_log_path=None):
    """One frozen single-T4 load, with safety checks and no diagnostic forward."""
    import torch
    from numerical_runtime import verify_quantized_compute_dtype
    from submission_runtime import ImageTeacher

    if torch.cuda.device_count() != 1 or 1 not in config['allowed_gpu_count']:
        raise ContractError('Submission model load requires exactly one visible T4')
    before = _free_memory_guard(torch, config['minimum_free_gpu_gib_before_load'])
    attempt = {'status': 'model_load_started', 'placement': 'single_gpu',
               'model_load_started': True, 'free_vram_before_attempt_gib': before}

    def persist():
        if attempt_log_path is not None:
            atomic_json(Path(attempt_log_path), {'attempts': [attempt]})

    persist()
    started = time.monotonic()
    try:
        # The constructor owns the original NF4/FP16 recipe, adapter attachment,
        # model type, device map, frozen weights, microbatch, normalization policy,
        # answer tokenization, and dtype inventory. No frozen method is patched.
        teacher = ImageTeacher(config, series, data_dir, base_dir, adapter_dir, 'single_gpu')
        load_seconds = time.monotonic() - started
        quantized = verify_quantized_compute_dtype(teacher.model)
        reserve = _free_memory_guard(torch, config['minimum_free_gpu_gib_after_diagnostic'])
        attempt.update(status='complete', load_seconds=load_seconds,
                       hf_device_map=teacher.device_map, dtype_report=teacher.dtype_report,
                       numerical_policy=teacher.numerical_policy, vision_runtime=teacher.vision_runtime,
                       observed_quantized_compute_dtypes=quantized,
                       free_vram_after_load_gib=reserve, diagnostic_performed=False)
        persist()
        return teacher, load_seconds, attempt
    except BaseException as error:
        attempt.update(status='failed', error_type=type(error).__name__, error=str(error),
                       traceback=traceback.format_exc(), elapsed_seconds=time.monotonic() - started)
        persist()
        raise


def _validate_score_records(records, shard):
    expected = [(uid, label) for uid in shard for label in LABELS]
    if [(row.get('study_id'), row.get('label')) for row in records] != expected:
        raise ContractError('Expected exactly twelve ordered target forwards per study')
    for row in records:
        if not all(math.isfinite(float(row[key])) for key in
                   ('native_no_logit', 'native_yes_logit', 'yes_probability')):
            raise ContractError('Nonfinite actual target score evidence')
        if not 0 <= row['yes_probability'] <= 1:
            raise ContractError('Actual target probability is outside [0, 1]')


def _empty_worker_result(directory):
    output = directory / 'inference'
    output.mkdir(exist_ok=False)
    pd.DataFrame(columns=['StudyInstanceUID', *LABELS]).to_csv(output / 'submission.csv', index=False)
    pd.DataFrame(columns=['StudyInstanceUID', 'study_seconds']).to_csv(output / 'study_timings.csv', index=False)
    result = {'status': 'complete', 'planned_studies': 0, 'completed_studies': 0,
              'model_loaded': False, 'total_seconds': 0.0}
    atomic_json(output / 'status.json', result)
    cache = {'policy': CACHE_POLICY, 'hits': 0, 'misses': 0, 'underlying_vision_calls': 0,
             'active_study': None, 'cache_released': True, 'studies': [],
             'decoder_forwards_per_study': 12}
    return result, cache


def worker(request_path, worker_id, worker_dir):
    """Import Torch only after this interpreter has one assigned visible GPU."""
    directory = Path(worker_dir)
    request = json.loads(Path(request_path).read_text())
    receipt_path = directory / 'worker_result.json'
    if receipt_path.exists():
        print('Worker receipt already exists; refusing overwrite', file=sys.stderr, flush=True)
        return 1
    receipt = {'worker_id': worker_id, 'pid': os.getpid(), 'phase': 'prepare',
               'status': 'running', 'placement': 'single_gpu', 'model_loaded': False,
               'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
    atomic_json(receipt_path, receipt)
    teacher = None
    try:
        if worker_id not in (0, 1) or receipt['visible_devices'] != str(worker_id):
            raise ContractError('Worker CUDA visibility must exactly match its replica index')
        verify_v8_sources(request['config_path'])
        verify_cache_source(request)
        config = read_config(Path(request['config_path']))
        test, series, sample = validate_tables(Path(request['data_dir']))
        shard = deterministic_shards(test)[worker_id]
        receipt['study_ids'] = shard
        import torch
        receipt['gpu_count'] = torch.cuda.device_count()
        if receipt['gpu_count'] != 1:
            raise ContractError('A replica must see exactly one GPU')
        prop = torch.cuda.get_device_properties(0)
        device = {'name': torch.cuda.get_device_name(0),
                  'capability_major': prop.major, 'capability_minor': prop.minor,
                  'total_gib': prop.total_memory / 2**30}
        validate_hardware([device], config)
        receipt['visible_gpu'] = device
        if not shard:
            result, cache_audit = _empty_worker_result(directory)
            receipt.update(phase='inference', load_seconds=0.0, score_record_count=0)
        else:
            from submission_runtime import run_inference
            from numerical_runtime import verify_quantized_compute_dtype
            from vision_cache import install_cached_teacher
            shard_test = test.set_index('StudyInstanceUID', drop=False).loc[shard].reset_index(drop=True)
            base, adapter, data = Path(request['base_dir']), Path(request['adapter_dir']), Path(request['data_dir'])
            teacher, load_seconds, metadata = load_submission_teacher(
                config, series, data, base, adapter, directory / 'model_attempt.json')
            atomic_json(directory / 'frozen_runtime.json', metadata)
            teacher = install_cached_teacher(teacher)
            receipt.update(phase='inference', load_seconds=load_seconds, model_loaded=True)
            atomic_json(receipt_path, receipt)
            _, result = run_inference(config, shard_test, series, sample, data, base, adapter,
                directory / 'inference', teacher, load_seconds, request['asset_verification_seconds'],
                request['dependency_install_seconds'], None)
            # Validate actual target forwards; these checks execute no extra inference.
            _validate_score_records(teacher.score_records, shard)
            receipt['score_record_count'] = len(teacher.score_records)
            receipt['observed_quantized_compute_dtypes'] = verify_quantized_compute_dtype(teacher.model)
            receipt['free_vram_after_inference_gib'] = _free_memory_guard(
                torch, config['minimum_free_gpu_gib_after_diagnostic'])
            cache_audit = teacher.vision_cache.summary()
            atomic_json(directory / 'score_logits.json', {'scores': teacher.score_records})
        _validate_cache(cache_audit, shard)
        atomic_json(directory / 'vision_cache.json', cache_audit)
        receipt['vision_cache'] = cache_audit
        path = directory / 'inference' / 'submission.csv'
        receipt.update(status='complete', inference_status=result, submission_path=str(path.resolve()),
                       submission_sha256=sha256_file(path))
        atomic_json(receipt_path, receipt)
        return 0
    except BaseException as error:
        cache = getattr(teacher, 'vision_cache', None)
        if cache is not None:
            cache.clear()
            atomic_json(directory / 'vision_cache_partial.json', cache.summary(collect_timing=False))
            atomic_json(directory / 'score_logits_partial.json', {'scores': teacher.score_records})
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
