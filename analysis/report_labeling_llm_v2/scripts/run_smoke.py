"""Plan offline; execution stays locked pending a separately reviewed GPU stage."""
import argparse
from datetime import datetime, timezone
import json
import os
import signal
import secrets
import stat
from pathlib import Path
import subprocess
import sys
import time

from core import LABELS, REPO, code_hashes, config, prompt_for, sha, validate_response, verify_protected
from v2_runtime import text_sha

from session_contract import ORDER, context_for, verify_child


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    path.chmod(0o600)


def append_json(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')
    stream.flush()
    os.fsync(stream.fileno())


def private_path(path):
    path = Path(path).resolve()
    if path.is_relative_to(REPO) and not path.is_relative_to(REPO/'state'):
        raise ValueError('Private artifacts in this checkout must remain under state/')
    return path


def checked_inputs(prepared):
    prepared = private_path(prepared)
    verify_protected()
    m = json.loads((prepared/'manifest.json').read_text())
    anchors = config('input_anchors')
    if (m['candidate_code_sha256'] != code_hashes() or
            m['source_sha256'] != anchors['source_sha256'] or
            m['split_sha256'] != anchors['split_sha256']):
        raise ValueError('Prepared version or original source/split differs')
    for name, digest in m['files'].items():
        if Path(name).name != name or sha(prepared/name) != digest:
            raise ValueError('Prepared artifact fingerprint mismatch')
    if sha(prepared/'inputs.jsonl') != config('runtime')['fixed_inputs_sha256']:
        raise ValueError('Inputs are not the original five development reports')
    rows = [json.loads(line) for line in (prepared/'inputs.jsonl').read_text().splitlines()]
    if len(rows) != 5 or len({r['StudyInstanceUID'] for r in rows}) != 5:
        raise ValueError('Exactly five unique development reports required')
    for row in rows:
        if (row.get('split') != 'development' or set(row)-{'StudyInstanceUID','Report','split','language','report_sha256'}
                or not isinstance(row['Report'],str) or not row['Report']
                or row.get('report_sha256') != text_sha(row['Report'])):
            raise ValueError('Only label-free development inputs are permitted')
    for model in ('medgemma','qwen'):
        saved = [json.loads(line) for line in (prepared/f'{model}_prompts.jsonl').read_text().splitlines()]
        expected = [dict(case_index=i,prompt=prompt_for(model,r['Report'])) for i,r in enumerate(rows)]
        if saved != expected:
            raise ValueError('Prepared prompts differ from reviewed candidate')
    return rows


def make_plan(prepared):
    prepared = private_path(prepared)
    rows = checked_inputs(prepared)
    return dict(status='DRY_PLAN_ONLY', synthetic=False, model_calls=0,
        tokenizer_preflight='NOT_RUN',
        prepared_manifest_sha256=sha(prepared/'manifest.json'),
        candidate_code_sha256=code_hashes(), input_sha256=sha(prepared/'inputs.jsonl'),
        unique_studies=5, maximum_primary_generations=20, run_order=ORDER,
        models={model:config(model) for model in ('medgemma','qwen')},
        runtime_policy=config('runtime'),
        cases=[dict(case_index=i,StudyInstanceUID=r['StudyInstanceUID'],language=r['language'],
                    report_sha256=text_sha(r['Report']),
                    prompt_sha256={m:text_sha(prompt_for(m,r['Report'])) for m in ('medgemma','qwen')})
               for i,r in enumerate(rows)])


def check_plan(prepared, plan_path, expected_sha):
    if sha(plan_path) != expected_sha:
        raise ValueError('Execution plan SHA does not match review')
    plan = json.loads(Path(plan_path).read_text())
    if plan != make_plan(prepared):
        raise ValueError('Execution plan is stale or changed')
    return plan


def execution_guard(execute):
    if not execute:
        raise PermissionError('Explicit --execute is required')
    if (not config('runtime')['gpu_execution_enabled'] or
            not config('policy')['gpu_execution_enabled'] or
            not all(config(m)['execution_enabled'] for m in ('medgemma','qwen'))):
        raise PermissionError('GPU execution is locked in this candidate; no model was loaded')


def run_once(prepared, plan, key, output, encoder_factory, backend_factory, synthetic=False, session_context=None, cache=None):
    """Dependency-injected engine. Real CLI adds guards and a parent process deadline."""
    if key not in ORDER:
        raise ValueError('Unknown model/repeat')
    if not synthetic and not session_context:
        raise PermissionError('Real runs require parent worker attestation')
    prepared, output = private_path(prepared), private_path(output)
    if not output.is_relative_to(prepared):
        raise ValueError('Run artifacts must be inside the private prepared package')
    if plan != make_plan(prepared):
        raise ValueError('Changed execution plan')
    rows = checked_inputs(prepared)
    output.mkdir(mode=0o700)  # Refuse overwrite/resume, including failed runs.
    model = key.rsplit('-',1)[0]
    started = time.perf_counter()
    manifest = dict(status='running', synthetic=synthetic, runtime_adapter_version=1, model_key=model,
        model_id=config(model)['model_id'], model_revision=config(model)['revision'],
        run_key=key, **(session_context or {}), started_at=datetime.now(timezone.utc).isoformat(),
        prepared_manifest_sha256=sha(prepared/'manifest.json'), candidate_code_sha256=code_hashes())
    write_json(output/'start_manifest.json',manifest)
    backend = None
    try:
        if not synthetic:
            if cache is None:
                raise ValueError('Explicit cache required for real generation')
            from load_preflight import audit_cache
            audit = audit_cache(cache, model)
            write_json(output/'cache_audit.json', audit)
            manifest['cache_audit_sha256'] = sha(output/'cache_audit.json')
            if not audit['complete']:
                raise ValueError('Pinned model cache is incomplete or corrupt')
        encoder = encoder_factory(model)
        encoded = [encoder.encode(prompt_for(model,r['Report'])) for r in rows]
        preflight = [dict(case_index=i,input_tokens=e.input_tokens,
                         rendered_prompt_sha256=text_sha(e.rendered_prompt)) for i,e in enumerate(encoded)]
        write_json(output/'tokenizer_preflight.json',dict(synthetic=synthetic,rows=preflight,metadata=encoder.metadata))
        if any(e.input_tokens > config(model)['max_input_tokens'] for e in encoded):
            raise ValueError('Context overflow in preflight; weights were not loaded')
        backend = backend_factory(encoder)
        write_json(output/'runtime_metadata.json',backend.metadata)
        if any(e.input_tokens + config(model)['max_new_tokens'] > backend.context_limit for e in encoded):
            raise ValueError('Input plus output budget exceeds model context')
        raw_path, prediction_path = output/'raw_generations.jsonl', output/'predictions.jsonl'
        complete = True
        with raw_path.open('x') as raw_stream, prediction_path.open('x') as parsed_stream:
            raw_path.chmod(0o600); prediction_path.chmod(0o600)
            for row, entry in zip(rows,encoded):
                generated = backend.generate(entry)
                record = dict(StudyInstanceUID=row['StudyInstanceUID'],synthetic=synthetic,
                    rendered_prompt=entry.rendered_prompt,rendered_prompt_sha256=text_sha(entry.rendered_prompt),
                    input_token_ids=entry.input_ids,**generated)
                # Durable original record before parser/normalizer is called.
                append_json(raw_stream,record)
                response = validate_response(record['raw_output'],row['Report'],record['generation_status'])
                append_json(parsed_stream,dict(record,response=response))
                if record['generation_status'] != 'completed':
                    complete = False
                    break  # Never repair, silently retry, or continue after a failed generation.
        manifest.update(status='completed' if complete else 'failed',
                        failure_reason=None if complete else 'noncompleted_generation',
                        predictions_sha256=sha(prediction_path),raw_generations_sha256=sha(raw_path))
    except Exception as error:
        # Never log arbitrary exceptions that may include private paths or credentials.
        manifest.update(status='failed',error_type=type(error).__name__)
    finally:
        manifest.update(runtime_seconds=time.perf_counter()-started,
                        peak_gpu_allocated_gib=backend.peak_gib() if backend else None,
                        finished_at=datetime.now(timezone.utc).isoformat())
        manifest['artifact_sha256'] = {p.name:sha(p) for p in output.iterdir() if p.is_file()}
        write_json(output/'run_manifest.json',manifest)
    return manifest['status']=='completed'


def supervise(command, timeout, launcher=subprocess.Popen, pass_fds=()):
    """Own a process group so timeout/interruption also stops child descendants."""
    process = launcher(command, start_new_session=True, pass_fds=pass_fds)
    try:
        code = process.wait(timeout=timeout)
        return code == 0, 'completed' if code == 0 else 'child_failed'
    except subprocess.TimeoutExpired:
        try: os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError: pass
        process.wait()
        return False, 'hard_timeout'
    except BaseException:
        # Do not leave inference running if the operator interrupts this parent.
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise


def worker_context(prepared, plan_sha, key, fd):
    if fd is None or fd < 3 or not stat.S_ISFIFO(os.fstat(fd).st_mode):
        raise PermissionError('Worker requires an inherited parent pipe')
    try:
        with os.fdopen(fd) as stream:
            message=json.loads(stream.read(8192))
    except (ValueError,OSError) as error:
        raise PermissionError('Invalid parent attestation') from error
    root=Path(prepared)/'adapter_runs'
    start_path=root/'session_start.json';start=json.loads(start_path.read_text());start_sha=sha(start_path)
    index=ORDER.index(key);dispatch_path=root/f'dispatch-{index}.json'
    dispatch=json.loads(dispatch_path.read_text())
    if (os.getppid()!=start['parent_pid'] or message.get('parent_pid')!=start['parent_pid'] or
            message.get('nonce') is None or text_sha(message['nonce'])!=dispatch['nonce_sha256'] or
            message.get('run_key')!=key or dispatch['run_key']!=key or dispatch['run_order_index']!=index or
            dispatch['session_id']!=start['session_id'] or dispatch['reviewed_plan_sha256']!=plan_sha or
            start['reviewed_plan_sha256']!=plan_sha or start['run_order']!=ORDER or
            dispatch['parent_session_start_sha256']!=start_sha or (root/'session_manifest.json').exists()):
        raise PermissionError('Worker was not dispatched by this active reviewed parent')
    for previous in range(index):
        verify_child(root,ORDER[previous],previous,start,start_sha)
    return context_for(start,start_sha,dispatch,sha(dispatch_path))


def execute_plan(prepared, plan_path, expected_sha, cache):
    execution_guard(True)
    check_plan(prepared,plan_path,expected_sha)
    from v2_runtime import check_versions
    check_versions()
    root=private_path(prepared)/'adapter_runs';root.mkdir(mode=0o700)
    started=time.perf_counter()
    session=dict(status='running',session_id=secrets.token_hex(16),parent_pid=os.getpid(),
                 reviewed_plan_sha256=expected_sha,plan_file=Path(plan_path).name,run_order=ORDER)
    write_json(root/'session_start.json',session);start_sha=sha(root/'session_start.json')
    results=[]
    for index,key in enumerate(ORDER):
        remaining=3600-(time.perf_counter()-started)
        if remaining<=0:
            results.append(dict(run_key=key,run_order_index=index,status='session_deadline'));break
        nonce=secrets.token_hex(32)
        dispatch=dict(session_id=session['session_id'],reviewed_plan_sha256=expected_sha,
            parent_session_start_sha256=start_sha,run_order_index=index,run_key=key,nonce_sha256=text_sha(nonce))
        write_json(root/f'dispatch-{index}.json',dispatch)
        reader,writer=os.pipe()
        try:
            os.write(writer,json.dumps(dict(parent_pid=os.getpid(),run_key=key,nonce=nonce)).encode())
        finally:os.close(writer)
        command=[sys.executable,str(Path(__file__).resolve()),'_worker',
                 '--prepared',str(prepared),'--plan',str(plan_path),'--plan-sha256',expected_sha,
                 '--cache',str(cache),'--run-key',key,'--worker-fd',str(reader),'--execute']
        try:
            ok,status=supervise(command,min(config('runtime')['hard_process_timeout_seconds'],remaining),pass_fds=(reader,))
        finally:os.close(reader)
        receipt=dict(run_key=key,run_order_index=index,status=status)
        if ok:
            try:
                verify_child(root,key,index,session,start_sha)
                receipt['run_manifest_sha256']=sha(root/key/'run_manifest.json')
            except (ValueError,KeyError,OSError):
                ok=False;receipt['status']='invalid_child_manifest'
        results.append(receipt)
        if not ok:break
    elapsed=time.perf_counter()-started
    complete=len(results)==4 and all(r['status']=='completed' for r in results) and elapsed<=3600
    write_json(root/'session_manifest.json',dict(status='completed' if complete else 'failed',
        session_id=session['session_id'],reviewed_plan_sha256=expected_sha,parent_session_start_sha256=start_sha,
        results=results,elapsed_seconds=elapsed,provider_stopped=False,
        provider_billing_stop='external operator required'))
    return complete


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['plan','run','_worker'])
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--plan-sha256')
    p.add_argument('--cache',type=Path)
    p.add_argument('--run-key',choices=ORDER)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--worker-fd',type=int,help=argparse.SUPPRESS)
    a=p.parse_args()
    prepared=private_path(a.prepared);plan_path=private_path(a.plan)
    if plan_path.parent!=prepared:
        p.error('Plan must be directly inside the private prepared package')
    if a.action=='plan':
        write_json(plan_path,make_plan(prepared))
        print(json.dumps(dict(status='DRY_PLAN_ONLY',model_calls=0,studies=5,
                              maximum_primary_generations=20,plan_sha256=sha(plan_path))))
        return
    if a.action=='_worker' and a.worker_fd is None:
        raise PermissionError('Worker requires an inherited parent pipe; standalone invocation refused')
    execution_guard(a.execute)  # Must precede Transformers imports or cache access.
    if not a.plan_sha256 or not a.cache:p.error('Reviewed plan SHA and explicit cache are required')
    plan=check_plan(prepared,plan_path,a.plan_sha256)
    if a.action=='run':
        ok=execute_plan(prepared,plan_path,a.plan_sha256,a.cache)
    else:
        if not a.run_key:p.error('Worker requires a fixed run key')
        context=worker_context(prepared,a.plan_sha256,a.run_key,a.worker_fd)
        from v2_runtime import HFEncoder,HFBackend,check_versions
        check_versions()
        ok=run_once(prepared,plan,a.run_key,prepared/'adapter_runs'/a.run_key,
                    lambda name:HFEncoder(name,a.cache),lambda encoder:HFBackend(encoder,a.cache),session_context=context,cache=a.cache)
    raise SystemExit(0 if ok else 1)


if __name__=='__main__':
    main()
