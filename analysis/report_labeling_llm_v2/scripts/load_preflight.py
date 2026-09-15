"""Pinned-cache audit and bounded GPU load check. Never performs model inference."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import secrets
import stat
from pathlib import Path
import sys
import time

from core import code_hashes, config, prompt_for, sha
from run_smoke import checked_inputs, private_path, write_json, supervise

MODELS = ('medgemma', 'qwen')
SESSION_SECONDS = 1800
WORKER_SECONDS = 900


def digest_file(path, algorithm):
    size=path.stat().st_size
    h=hashlib.sha256() if algorithm=='sha256' else hashlib.sha1()
    if algorithm not in ('sha256','git_blob_sha1'):
        raise ValueError('Unsupported digest')
    if algorithm=='git_blob_sha1':
        h.update(b'blob '+str(size).encode()+b'\0')
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):
            h.update(chunk)
    return h.hexdigest()


def audit_cache(cache, model):
    expected=config('weight_manifest')[model]
    cfg=config(model)
    if expected['model_id']!=cfg['model_id'] or expected['revision']!=cfg['revision']:
        raise ValueError('Manifest revision differs from reviewed model')
    snapshot=Path(cache)/('models--'+cfg['model_id'].replace('/','--'))/'snapshots'/cfg['revision']
    records=[]
    for entry in expected['files']:
        name=entry['file']
        if Path(name).name!=name:
            raise ValueError('Nested or unsafe manifest path')
        path=snapshot/name
        present=path.is_file()
        size=path.stat().st_size if present else None
        digest=digest_file(path,entry['algorithm']) if present and size==entry['bytes'] else None
        records.append(dict(file=name,bytes=size,expected_bytes=entry['bytes'],
                            digest=digest,verified=digest==entry['digest']))
    weights={e['file'] for e in expected['files'] if e['file'].endswith('.safetensors')}
    if not weights or len({e['file'] for e in expected['files']})!=len(expected['files']):
        raise ValueError('Empty weight set or duplicate manifest entry')
    complete=all(r['verified'] for r in records)
    index=snapshot/'model.safetensors.index.json'
    if complete and index.exists():
        mapping=json.loads(index.read_text()).get('weight_map',{})
        if not mapping or set(mapping.values())!=weights:
            raise ValueError('Weight index does not cover exactly the expected shards')
    return dict(model=model,revision=cfg['revision'],complete=complete,files=records,
                required_weight_bytes=sum(e['bytes'] for e in expected['files'] if e['file'] in weights))


def make_plan(prepared):
    rows=checked_inputs(prepared)
    return dict(stage='LOAD_ONLY_NO_INFERENCE',models=list(MODELS),studies=len(rows),
                candidate_code_sha256=code_hashes(),
                prepared_manifest_sha256=sha(Path(prepared)/'manifest.json'),
                input_sha256=sha(Path(prepared)/'inputs.jsonl'),
                session_timeout_seconds=SESSION_SECONDS,worker_timeout_seconds=WORKER_SECONDS,
                generation_calls=0,training_steps=0,automatic_downloads=False,
                provider_billing_stop='operator must stop pod after completion/failure/deadline')


def check_plan(prepared,plan_path,plan_sha):
    if sha(plan_path)!=plan_sha or json.loads(Path(plan_path).read_text())!=make_plan(prepared):
        raise ValueError('Load plan is stale or differs from reviewed digest')


def load_model(prepared,model,cache,output,plan_path,plan_sha):
    """Called only in a bounded subprocess. Transfers one prompt at a time; no forward pass."""
    check_plan(prepared,plan_path,plan_sha)
    rows=checked_inputs(prepared)
    output=private_path(output);output.mkdir(mode=0o700)
    started=time.perf_counter()
    write_json(output/'start.json',dict(status='running',model=model,plan_sha256=plan_sha))
    try:
        cache_audit=audit_cache(cache,model)
        write_json(output/'cache_audit.json',cache_audit)
        if not cache_audit['complete']:
            raise ValueError('Pinned weight/config cache is incomplete or corrupt')
        from v2_runtime import HFEncoder, HFBackend, check_versions
        check_versions()
        encoder=HFEncoder(model,cache)
        entries=[encoder.encode(prompt_for(model,r['Report'])) for r in rows]
        if any(e.input_tokens>config(model)['max_input_tokens'] for e in entries):
            raise ValueError('Input budget exceeded before weight load')
        backend=HFBackend(encoder,cache)
        torch=backend.torch
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            raise ValueError('Native BF16 support is required')
        # Defense against accidental later edits adding a forward/generation call.
        def prohibited(*args,**kwargs):
            raise RuntimeError('Inference is forbidden in the load-only preflight')
        backend.model.generate=prohibited
        backend.model.forward=prohibited
        if any(p.device.type!='cuda' or (p.is_floating_point() and p.dtype!=torch.bfloat16)
               for p in backend.model.parameters()):
            raise ValueError('Parameters are not entirely CUDA/BF16')
        attention=getattr(backend.model.config,'_attn_implementation',None)
        if attention!='sdpa':
            raise ValueError('Loaded attention configuration is not SDPA')
        load_peak=backend.peak_gib()
        for e in entries:
            if e.input_tokens+config(model)['max_new_tokens']>backend.context_limit:
                raise ValueError('Loaded context budget exceeded')
            device_inputs={k:v.to('cuda:0') if hasattr(v,'to') else v for k,v in e.inputs.items()}
            torch.cuda.synchronize()
            if any(v.device.type!='cuda' for v in device_inputs.values() if hasattr(v,'device')):
                raise ValueError('Prompt tensors did not reach CUDA')
            del device_inputs
        torch.cuda.synchronize()
        result=dict(status='completed',model=model,plan_sha256=plan_sha,
            stage='LOAD_ONLY_NO_INFERENCE',candidate_code_sha256=code_hashes(),
            runtime=backend.metadata,attention_config=attention,
            input_tokens=[e.input_tokens for e in entries],
            load_peak_allocated_gib=load_peak,peak_allocated_gib=backend.peak_gib(),
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            device_free_gib=torch.cuda.mem_get_info()[0]/2**30,
            all_parameters_cuda_bf16=True,generation_calls=0,forward_calls=0,
            sdpa_kernel_execution_verified=False,kv_cache_memory_measured=False)
        checked_inputs(prepared)
    except Exception as error:
        result=dict(status='failed',model=model,plan_sha256=plan_sha,error_type=type(error).__name__)
    result.update(elapsed_seconds=time.perf_counter()-started,
                  completed_at=datetime.now(timezone.utc).isoformat())
    write_json(output/'result.json',result)
    return result['status']=='completed'


def run(args):
    if not args.execute_load:
        raise PermissionError('Explicit --execute-load is required; this may use paid GPU time')
    check_plan(args.prepared,args.plan,args.plan_sha256)
    output=private_path(args.output);output.mkdir(mode=0o700)
    start=time.monotonic()
    write_json(output/'session_start.json',dict(plan_sha256=args.plan_sha256,stage='LOAD_ONLY_NO_INFERENCE'))
    results=[]
    for model in MODELS:
        remaining=SESSION_SECONDS-(time.monotonic()-start)
        if remaining<=0:
            break
        nonce=secrets.token_hex(32)
        read_fd,write_fd=os.pipe()
        with os.fdopen(write_fd,'w') as stream:
            json.dump(dict(nonce=nonce,model=model,plan_sha256=args.plan_sha256),stream)
        command=[sys.executable,str(Path(__file__).resolve()),'_load-worker',
            '--prepared',str(args.prepared.resolve()),'--cache',str(args.cache.resolve()),
            '--plan',str(args.plan.resolve()),'--plan-sha256',args.plan_sha256,
            '--model',model,'--output',str(output/model),'--execute-load',
            '--attestation-fd',str(read_fd),'--parent-pid',str(os.getpid()),
            '--nonce-sha256',hashlib.sha256(nonce.encode()).hexdigest()]
        try:
            child_ok,reason=supervise(command,min(WORKER_SECONDS,remaining),pass_fds=(read_fd,))
        finally:
            os.close(read_fd)
        path=output/model/'result.json'
        receipt=json.loads(path.read_text()) if path.exists() else {}
        ok=(child_ok and receipt.get('status')=='completed' and receipt.get('model')==model
            and receipt.get('plan_sha256')==args.plan_sha256
            and receipt.get('candidate_code_sha256')==code_hashes())
        results.append(dict(model=model,completed=ok,process_status=reason,result_sha256=sha(path) if path.exists() else None))
        if not ok:break
    success=len(results)==2 and all(r['completed'] for r in results)
    write_json(output/'session_result.json',dict(status='completed' if success else 'failed',
        stage='LOAD_ONLY_NO_INFERENCE',plan_sha256=args.plan_sha256,models=results,
        elapsed_seconds=time.monotonic()-start,generation_calls=0,forward_calls=0,
        provider_stopped=False))
    return success


def verify_parent(args):
    if not args.execute_load or args.attestation_fd is None or args.parent_pid != os.getppid():
        raise PermissionError('Worker requires the live bounded parent')
    if not stat.S_ISFIFO(os.fstat(args.attestation_fd).st_mode):
        raise PermissionError('Worker requires an inherited anonymous pipe')
    with os.fdopen(args.attestation_fd) as stream:
        payload=json.loads(stream.read(2048))
    if (hashlib.sha256(payload['nonce'].encode()).hexdigest()!=args.nonce_sha256 or
        payload['model']!=args.model or payload['plan_sha256']!=args.plan_sha256):
        raise PermissionError('Parent attestation differs')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['plan','audit','run','_load-worker'])
    p.add_argument('--prepared',type=Path);p.add_argument('--cache',type=Path)
    p.add_argument('--plan',type=Path);p.add_argument('--plan-sha256')
    p.add_argument('--model',choices=MODELS);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute-load',action='store_true')
    p.add_argument('--attestation-fd',type=int);p.add_argument('--parent-pid',type=int)
    p.add_argument('--nonce-sha256')
    a=p.parse_args()
    if a.action=='plan':write_json(private_path(a.output),make_plan(a.prepared))
    elif a.action=='audit':write_json(private_path(a.output),audit_cache(a.cache,a.model))
    elif a.action=='run':sys.exit(0 if run(a) else 1)
    else:
        verify_parent(a)
        sys.exit(0 if load_model(a.prepared,a.model,a.cache,a.output,a.plan,a.plan_sha256) else 1)


if __name__=='__main__':main()
