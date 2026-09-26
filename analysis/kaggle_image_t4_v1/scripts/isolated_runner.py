"""Isolate a failed CUDA constructor before the single allowed dual-T4 attempt."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from inference_core import ContractError, atomic_json, read_config, validate_free_memory

OOM_EXIT = 42


def _worker_command(request_path, placement, attempt_dir):
    # exec a new interpreter; never fork an initialized CUDA interpreter.
    return [sys.executable, '-u', str(Path(__file__).resolve()), '--worker',
            str(request_path), '--placement', placement, '--attempt-dir', str(attempt_dir)]


def _kill_and_reap(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=15)


def run_isolated(request_path: Path, deadline_monotonic: float, cuda=None) -> dict:
    """One shared deadline; successful worker performs the whole example once."""
    if cuda is None:
        import torch
        cuda = torch.cuda
    request_path = Path(request_path).resolve()
    request = json.loads(request_path.read_text())
    config = read_config(Path(request['config_path']))
    count = cuda.device_count()
    if count not in config['allowed_gpu_count']:
        raise ContractError('Visible T4 count is outside frozen scope')
    output_root = Path(request['output_root'])
    output_root.mkdir(parents=True, exist_ok=False)
    attempts = []

    def persist():
        atomic_json(output_root / 'placement_attempts.json', {'attempts': attempts})

    def memory():
        for i in range(count):
            cuda.synchronize(i)
        return [cuda.mem_get_info(i)[0] / 2**30 for i in range(count)]

    for placement in (['single_gpu', 'two_gpu'] if count == 2 else ['single_gpu']):
        before = memory()
        validate_free_memory(before[:1 if placement == 'single_gpu' else 2],
                             config['minimum_free_gpu_gib_before_load'])
        remaining = deadline_monotonic - time.monotonic()
        if remaining <= 0:
            raise ContractError('Shared inference deadline expired; no worker started')
        attempt_dir = output_root / placement
        attempt_dir.mkdir(exist_ok=False)
        row = {'placement': placement, 'status': 'worker_starting',
               'free_vram_before_attempt_gib': before}
        attempts.append(row)
        persist()
        print('T4_WORKER_START', json.dumps(row), flush=True)
        process = subprocess.Popen(_worker_command(request_path, placement, attempt_dir),
                                   start_new_session=True)
        row['process_pid'] = process.pid
        row['status'] = 'worker_running'
        persist()
        try:
            returncode = process.wait(timeout=max(0.01, deadline_monotonic-time.monotonic()))
        except BaseException as error:
            _kill_and_reap(process)
            row.update(status='timeout' if isinstance(error, subprocess.TimeoutExpired) else 'interrupted',
                       returncode=process.returncode, process_reaped=True)
            persist()
            raise ContractError('T4 worker stopped and reaped at shared deadline/interruption; no fallback') from error
        # wait() has reaped the child. Only now inspect global GPU free memory.
        after = memory()
        row.update(returncode=returncode, process_reaped=True,
                   free_vram_after_process_exit_gib=after, status='worker_exited')
        persist()
        receipt_path = attempt_dir / 'worker_result.json'
        try:
            result = json.loads(receipt_path.read_text())
        except (OSError, ValueError) as error:
            raise ContractError('Worker result missing/invalid; no fallback') from error
        row['worker_result'] = result
        row['status'] = result.get('status', 'invalid')
        persist()
        print('T4_WORKER_EXIT', json.dumps(row), flush=True)
        if result.get('placement') != placement or result.get('pid') != process.pid:
            raise ContractError('Worker result identity mismatch; no fallback')
        if returncode == 0 and result.get('status') == 'complete' and result.get('phase') == 'inference':
            return {'attempts': attempts, 'result': result, 'output_root': str(output_root)}
        is_prepare_oom = (returncode == OOM_EXIT and result.get('status') == 'cuda_oom'
                          and result.get('phase') == 'prepare')
        if not is_prepare_oom or placement != 'single_gpu' or count != 2:
            raise ContractError(f"{placement} failed: {result.get('error_type')}: {result.get('error')}; no fallback")
        if abs(after[0]-before[0]) > 0.5:
            raise ContractError(f'CUDA cleanup leak after worker exit: {after[0]:.2f} vs {before[0]:.2f} GiB; refusing dual-T4 attempt')
        validate_free_memory(after, config['minimum_free_gpu_gib_before_load'])
        print('T4_CLEANUP_RECOVERED', json.dumps({'before_gib':before, 'after_process_exit_gib':after}), flush=True)
    raise ContractError('No permitted T4 placement completed')


def worker(request_path, placement, attempt_dir):
    import torch
    from inference_core import sha256_file, validate_tables
    from submission_runtime import prepare_t4_teacher, run_inference

    request = json.loads(Path(request_path).read_text())
    attempt_dir = Path(attempt_dir)
    receipt = {'placement':placement, 'pid':os.getpid(), 'phase':'prepare', 'status':'running'}
    receipt_path = attempt_dir/'worker_result.json'
    atomic_json(receipt_path, receipt)
    config = read_config(Path(request['config_path']))
    data = Path(request['data_dir'])
    test, series, sample = validate_tables(data)
    base, adapter = Path(request['base_dir']), Path(request['adapter_dir'])
    attempt_log = attempt_dir/'model_attempt.json'
    try:
        teacher, load_seconds, diagnostic, attempts = prepare_t4_teacher(
            config, series, data, base, adapter, str(test.StudyInstanceUID.iloc[0]),
            attempt_log, only_placement=placement)
        frozen = {'runtime_variant':config['runtime_variant'], 'placement':teacher.placement,
                  'hf_device_map':teacher.device_map, 'dtype_report':teacher.dtype_report,
                  'load_seconds':load_seconds, 'diagnostic':diagnostic, 'attempts':attempts}
        frozen_json = json.dumps(frozen, sort_keys=True, indent=2)
        (attempt_dir/'t4_configuration_frozen.json').write_text(frozen_json)
        print('FROZEN_T4_CONFIG_SHA256',hashlib.sha256(frozen_json.encode()).hexdigest(),flush=True)
        print('FROZEN_T4_CONFIG',json.dumps(frozen),flush=True)
        receipt['phase'] = 'inference'
        atomic_json(receipt_path,receipt)
        out = attempt_dir/'example'
        prediction,status = run_inference(config,test,series,sample,data,base,adapter,out,
            teacher,load_seconds,request['asset_verification_seconds'],
            request['dependency_install_seconds'],diagnostic)
        receipt.update(status='complete',example_status=status,
                       submission_path=str(out/'submission.csv'),
                       submission_sha256=sha256_file(out/'submission.csv'))
        atomic_json(receipt_path,receipt)
        print('T4_EXAMPLE_COMPLETE',json.dumps(receipt),flush=True)
        return 0
    except Exception as error:
        receipt.update(status='failed',error_type=type(error).__name__,error=str(error))
        exit_code = 1
        if receipt['phase'] == 'prepare' and attempt_log.is_file():
            records = json.loads(attempt_log.read_text()).get('attempts',[])
            receipt['model_attempts'] = records
            if len(records) == 1 and records[0].get('placement') == placement and records[0].get('status') == 'oom':
                receipt.update(status='cuda_oom',original_error_type=records[0]['error_type'],
                               original_error=records[0]['error'])
                exit_code = OOM_EXIT
        receipt['traceback'] = traceback.format_exc()
        atomic_json(receipt_path,receipt)
        print('T4_WORKER_FAILURE',json.dumps(receipt),flush=True)
        return exit_code


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker',type=Path,required=True)
    parser.add_argument('--placement',choices=['single_gpu','two_gpu'],required=True)
    parser.add_argument('--attempt-dir',type=Path,required=True)
    args = parser.parse_args()
    raise SystemExit(worker(args.worker,args.placement,args.attempt_dir))
