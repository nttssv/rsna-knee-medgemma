"""Bounded v3 execution revision. Real generation requires a current private grant."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import stat
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
V3 = ROOT.parent/'report_labeling_llm_v3'
V2 = ROOT.parent/'report_labeling_llm_v2'
spec = importlib.util.spec_from_file_location('runtime_frozen_v3', V3/'scripts/v3_candidate.py')
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)
core = candidate.core
sys.path.insert(0, str(V2/'scripts'))
from run_smoke import supervise  # Frozen process-group kill/reap implementation.

ORDER = candidate.experiment()['run_order']
sha = core.sha


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    with Path(path).open('x') as stream:
        os.chmod(path,0o600)
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())


def append(stream, value):
    stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
    stream.flush();os.fsync(stream.fileno())


def read(path):
    return json.loads(Path(path).read_text(),object_pairs_hook=core.no_duplicates)


def lines(path):
    return [json.loads(line,object_pairs_hook=core.no_duplicates) for line in Path(path).read_text().splitlines()]


def private(path):
    path=Path(path).resolve()
    if path.is_relative_to(REPO) and not path.is_relative_to(REPO/'state'):
        raise ValueError('Use ignored state/ or an external private directory')
    return path


def policy():
    return read(ROOT/'configs/runtime.json')


def verify_history():
    files=read(ROOT/'configs/protected_history.json')['files']
    for name,expected in files.items():
        if sha(REPO/name)!=expected:raise ValueError('Protected historical file changed: '+name)
    candidate.verify_history()
    return len(files)


def fingerprints():
    paths=[ROOT/'PROTOCOL.md']
    paths += [p for folder in ('scripts','configs','tests') for p in (ROOT/folder).rglob('*')
              if p.is_file() and '__pycache__' not in p.parts]
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}


def model_config():
    """Explicit copy; never mutate frozen v2 module/configuration globals."""
    cfg=core.config('medgemma').copy(); new=candidate.experiment()
    for key in ('model_id','revision','dtype','quantization','adapter','batch_size','attn_implementation',
                'do_sample','num_beams','seed','max_input_tokens','max_new_tokens','max_time_seconds'):
        cfg[key]=new[key]
    cfg.update(prompt_version='v3_arm_specific',prompt_file=None,execution_enabled=False,status='UNEXECUTED_RUNTIME')
    if new['required_versions']!=core.config('runtime')['required_versions']:
        raise ValueError('Pinned versions differ')
    return cfg


def parse_time(value):
    result=datetime.fromisoformat(value)
    if result.tzinfo is None:raise ValueError('Timezone-aware timestamp required')
    return result


def verify_watchdog(path,grant):
    receipt=read(private(path))
    if (receipt['status']!='armed' or receipt.get('cli_syntax_and_read_access_verified') is not True or receipt['pod_id']!=grant['pod_id'] or
        receipt['deadline']!=grant['provider_deadline'] or receipt['script_sha256']!=sha(ROOT/'scripts/stop_watchdog.py') or
        os.environ.get('RUNPOD_POD_ID')!=grant['pod_id'] or type(receipt['pid']) is not int or receipt['pid']<=1):
        raise PermissionError('Missing or mismatched self-stop watchdog')
    os.kill(receipt['pid'],0)
    cmdline=Path('/proc')/str(receipt['pid'])/'cmdline'
    if not cmdline.exists() or str(ROOT/'scripts/stop_watchdog.py').encode() not in cmdline.read_bytes():
        raise PermissionError('Watchdog process identity is unverified')
    return receipt


def guard(execute=False,prepared=None,expected_sha=None,authorization=None,watchdog=None,current=None):
    p=policy()
    if not execute or not p['execution_enabled'] or not all((prepared,expected_sha,authorization,watchdog)):
        raise PermissionError('GPU execution disabled: explicit execution, reviewed plan, private authorization and live watchdog are required')
    plan,_=checked_plan(prepared,expected_sha)
    if plan['synthetic']:raise PermissionError('Synthetic plan cannot authorize real execution')
    authorization=private(authorization)
    if authorization.stat().st_mode & 0o077:raise PermissionError('Authorization must be private (0600)')
    grant=read(authorization);proposal=read(ROOT/'configs/resource_proposal.json')
    required={'approved','approval_reference','plan_sha256','proposal_sha256','approved_at','provider_start_requested_at',
        'provider_deadline','maximum_usd','actual_compute_usd_per_hour','actual_storage_usd_per_hour','pod_id','gpu_name','gpu_count'}
    if set(grant)!=required or grant['approved'] is not True or not str(grant['approval_reference']).strip():
        raise PermissionError('Explicit user approval record required')
    if (grant['plan_sha256']!=expected_sha or grant['proposal_sha256']!=sha(ROOT/'configs/resource_proposal.json') or
        grant['gpu_name']!=proposal['gpu_name'] or type(grant['gpu_count']) is not int or grant['gpu_count']!=1 or not grant['pod_id']):
        raise PermissionError('Approval does not cover this plan/resource')
    current=current or datetime.now(timezone.utc)
    if current>parse_time(proposal['quote_valid_until']):
        raise PermissionError('Resource proposal expired; refresh the quote and reviewed plan')
    approved=parse_time(grant['approved_at']);began=parse_time(grant['provider_start_requested_at']);end=parse_time(grant['provider_deadline'])
    seconds=(end-began).total_seconds()
    if not approved<=began<=current<end or not 0<seconds<=p['maximum_provider_seconds']:
        raise PermissionError('Approval is stale or provider deadline exceeds the approved window')
    amounts=[grant[k] for k in ('maximum_usd','actual_compute_usd_per_hour','actual_storage_usd_per_hour')]
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in amounts):
        raise PermissionError('Invalid cost bounds')
    if (not 0<grant['maximum_usd']<=p['maximum_approved_usd'] or
        grant['actual_compute_usd_per_hour']>proposal['compute_usd_per_hour'] or
        grant['actual_storage_usd_per_hour']>proposal['storage_usd_per_hour_estimate'] or
        sum(amounts[1:])*seconds/3600>grant['maximum_usd']):
        raise PermissionError('Actual quoted charges exceed the reviewed budget/rate')
    if (end-current).total_seconds()<=p['reserved_copy_stop_seconds']:
        raise PermissionError('Insufficient time before the copy-and-stop reserve')
    verify_watchdog(watchdog,grant)
    return grant


def prepare(output, source=None, state=None, synthetic=False):
    verify_history(); output=private(output)
    if output.exists():raise FileExistsError('Never overwrite a prepared experiment')
    if synthetic and source is not None:raise ValueError('Demo cannot consume competition inputs')
    if not synthetic and source is None:raise ValueError('Fixed candidate package required')
    output.mkdir(parents=True,mode=0o700)
    if synthetic:
        rows=[dict(StudyInstanceUID='synthetic-'+str(i),Report='Synthetic software fixture '+str(i)+'.',
                   split='development',language='synthetic') for i in range(5)]
        for row in rows:row['report_sha256']=hashlib.sha256(row['Report'].encode()).hexdigest()
        with (output/'inputs.jsonl').open('x') as f:
            os.chmod(f.name,0o600)
            for row in rows:append(f,row)
    else:
        source=private(source); state=private(state)
        if sha(source/'plan.json')!=policy()['candidate_plan_sha256']:
            raise ValueError('Wrong frozen candidate plan')
        old=read(source/'plan.json')
        if old['candidate_sha256']!=candidate.fingerprints():raise ValueError('Candidate changed')
        for name,expected in old['files'].items():
            if Path(name).name!=name or sha(source/name)!=expected:raise ValueError('Candidate artifact changed')
            shutil.copyfile(source/name,output/name);(output/name).chmod(0o600)
        shutil.copyfile(source/'plan.json',output/'candidate_plan.json')
        split=state/'runs/report-labeling-20260913-v1/splits.csv'
        if sha(split)!=old['split_sha256']:raise ValueError('Frozen split changed')
        shutil.copyfile(split,output/'splits.csv')
        for name in ('candidate_plan.json','splits.csv'):(output/name).chmod(0o600)
    files={p.name:sha(p) for p in output.iterdir() if p.is_file()}
    plan=dict(stage='SYNTHETIC_SOFTWARE_DEMO' if synthetic else 'LOCAL_RUNTIME_PLAN_NOT_RUN',
        synthetic=synthetic,model_calls=0,files=files,runtime_sha256=fingerprints(),
        candidate_sha256=candidate.fingerprints(),policy=policy(),model=model_config(),
        run_order=ORDER,studies=5,maximum_generations=20,planned_condition_cells=240)
    write(output/'plan.json',plan)
    checked_plan(output,sha(output/'plan.json'))
    return dict(synthetic=synthetic,plan_sha256=sha(output/'plan.json'),model_calls=0)


def checked_plan(prepared,expected_sha):
    prepared=private(prepared); verify_history()
    if sha(prepared/'plan.json')!=expected_sha:raise ValueError('Reviewed runtime plan changed')
    plan=read(prepared/'plan.json')
    if (plan['runtime_sha256']!=fingerprints() or plan['candidate_sha256']!=candidate.fingerprints()
        or plan['policy']!=policy() or plan['model']!=model_config() or plan['run_order']!=ORDER
        or plan['studies']!=5 or plan['maximum_generations']!=20 or plan['planned_condition_cells']!=240):
        raise ValueError('Runtime plan is stale or outside the fixed design')
    for name,expected in plan['files'].items():
        if Path(name).name!=name or sha(prepared/name)!=expected:raise ValueError('Prepared artifact changed')
    rows=lines(prepared/'inputs.jsonl')
    if len(rows)!=5 or len({r['StudyInstanceUID'] for r in rows})!=5:raise ValueError('Expected five unique studies')
    for row in rows:
        if (set(row)!={'StudyInstanceUID','Report','split','language','report_sha256'} or
            row['split']!='development' or not row['Report'] or
            hashlib.sha256(row['Report'].encode()).hexdigest()!=row['report_sha256']):
            raise ValueError('Unexpected input fields or fingerprint')
    if plan['synthetic'] is True:
        if plan['files'].keys()!={'inputs.jsonl'} or any(r['Report']!='Synthetic software fixture '+str(i)+'.'
            or r['StudyInstanceUID']!='synthetic-'+str(i) or r['language']!='synthetic' for i,r in enumerate(rows)):
            raise ValueError('Demo is limited to the five fabricated fixtures')
    elif plan['synthetic'] is False:
        old=read(prepared/'candidate_plan.json')
        if (sha(prepared/'candidate_plan.json')!=policy()['candidate_plan_sha256'] or
            old['candidate_sha256']!=candidate.fingerprints() or
            sha(prepared/'inputs.jsonl')!=candidate.experiment()['source_inputs_sha256'] or
            sha(prepared/'splits.csv')!=old['split_sha256']):raise ValueError('Unanchored real inputs')
        if set(plan['files'])!=set(old['files'])|{'candidate_plan.json','splits.csv'}:
            raise ValueError('Incomplete candidate package')
        if any(sha(prepared/name)!=expected for name,expected in old['files'].items()):
            raise ValueError('Frozen candidate files changed')
        with (prepared/'splits.csv').open() as f:assignments=list(csv.DictReader(f))
        dev={r['StudyInstanceUID'] for r in assignments if r['split']=='development'}
        if len(assignments)!=58 or len({r['StudyInstanceUID'] for r in assignments})!=58 or len(dev)!=40 or sum(r['split']=='validation' for r in assignments)!=18 or not {r['StudyInstanceUID'] for r in rows}<=dev:
            raise ValueError('Expected unchanged 40/18 split')
        for arm in ('control','candidate'):
            if lines(prepared/(arm+'_prompts.jsonl'))!=[dict(case_index=i,prompt=candidate.make_prompt(arm,r['Report'])) for i,r in enumerate(rows)]:
                raise ValueError('Prompt changed')
    else:raise ValueError('Synthetic flag must be boolean')
    return plan,rows


class SyntheticEncoder:
    metadata={'synthetic':True,'tokenizer':'none; fabricated software fixture'}
    def encode(self,prompt):
        from v2_runtime import Encoded
        return Encoded(prompt,None,[7])


class SyntheticBackend:
    context_limit=131072
    metadata={'synthetic':True,'gpu_name':None,'model_loaded':False}
    def generate(self,encoded):
        raw=json.dumps({c:dict(label='not_mentioned',evidence_text='',confidence=0) for c in core.LABELS})
        return dict(raw_output=raw,decoded_with_special_tokens=raw,output_token_ids=[8,1],input_tokens=1,
                    output_tokens=2,runtime_seconds=0.0,generation_status='completed')
    def peak_gib(self):return 0.0


def real_encoder(cache):
    from v2_runtime import HFEncoder,check_versions
    check_versions()
    encoder=HFEncoder('medgemma',cache);encoder.cfg=model_config()
    return encoder


def real_backend(encoder,cache):
    from v2_runtime import HFBackend
    backend=HFBackend(encoder,cache)
    if backend.metadata['gpu_name']!=read(ROOT/'configs/resource_proposal.json')['gpu_name']:
        raise ValueError('Actual GPU differs from approved hardware')
    if not backend.torch.cuda.is_bf16_supported(including_emulation=False):raise ValueError('Native BF16 required')
    if any(p.device.type!='cuda' or (p.is_floating_point() and p.dtype!=backend.torch.bfloat16) for p in backend.model.parameters()):
        raise ValueError('Model must be entirely CUDA/BF16')
    if getattr(backend.model.config,'_attn_implementation',None)!='sdpa':raise ValueError('SDPA required')
    backend.metadata.update(effective_v3_model_config=encoder.cfg,
        nvidia_driver_version=__import__('subprocess').check_output(
            ['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip())
    return backend


def context(prepared,key):
    root=Path(prepared)/'session';start=read(root/'start.json');index=ORDER.index(key)
    dispatch=read(root/f'dispatch-{index}.json')
    if (dispatch['key']!=key or dispatch['index']!=index or dispatch['session_id']!=start['session_id']
        or dispatch['start_sha256']!=sha(root/'start.json') or dispatch['plan_sha256']!=start['plan_sha256']):
        raise ValueError('Dispatch is not bound to parent')
    return dict(key=key,index=index,session_id=start['session_id'],plan_sha256=start['plan_sha256'],
                start_sha256=sha(root/'start.json'),dispatch_sha256=sha(root/f'dispatch-{index}.json'))


def verify_parent(prepared,key,fd):
    if fd is None or fd<3 or not stat.S_ISFIFO(os.fstat(fd).st_mode):raise PermissionError('Inherited parent pipe required')
    with os.fdopen(fd) as f:payload=json.loads(f.read(8192))
    root=Path(prepared)/'session';start=read(root/'start.json')
    dispatch=read(root/f'dispatch-{ORDER.index(key)}.json')
    if (payload.get('parent_pid')!=os.getppid() or start['parent_pid']!=os.getppid() or
        payload.get('key')!=key or payload.get('plan_sha256')!=start['plan_sha256'] or
        digest(payload.get('nonce'))!=dispatch['nonce_sha256'] or (root/'result.json').exists()):
        raise PermissionError('Worker lacks live reviewed parent provenance')
    for previous in ORDER[:ORDER.index(key)]:verify_run(prepared,previous)
    return context(prepared,key)


def run_once(prepared,expected_sha,key,cache=None,worker_fd=None,execute=False,authorization=None,watchdog=None):
    plan,rows=checked_plan(prepared,expected_sha)
    if key not in ORDER:raise ValueError('Unknown run')
    if not plan['synthetic']:
        guard(execute,prepared,expected_sha,authorization,watchdog)
        start=read(Path(prepared)/'session/start.json')
        if start['authorization_sha256']!=sha(authorization) or start['watchdog_sha256']!=sha(watchdog):
            raise PermissionError('Authorization/watchdog changed after parent dispatch')
    ctx=verify_parent(prepared,key,worker_fd)
    if ctx['plan_sha256']!=expected_sha:raise ValueError('Parent plan mismatch')
    out=Path(prepared)/'session'/key;out.mkdir(mode=0o700)
    started=time.monotonic();receipt=dict(ctx,synthetic=plan['synthetic'],status='failed',started_at=now())
    try:
        if plan['synthetic']:
            encoder=SyntheticEncoder()
        else:
            from load_preflight import audit_cache
            audit=audit_cache(cache,'medgemma');write(out/'cache_audit.json',audit)
            if not audit['complete']:raise ValueError('Pinned full weight/config cache failed audit')
            encoder=real_encoder(cache)
        entries=[encoder.encode(candidate.make_prompt(key.split('-')[0],r['Report'])) for r in rows]
        write(out/'preflight.json',dict(encoder=encoder.metadata,input_tokens=[e.input_tokens for e in entries]))
        if any(e.input_tokens>plan['model']['max_input_tokens'] for e in entries):
            raise ValueError('Input budget exceeded before weight load')
        backend=SyntheticBackend() if plan['synthetic'] else real_backend(encoder,cache)
        if any(e.input_tokens+plan['model']['max_new_tokens']>backend.context_limit for e in entries):
            raise ValueError('Input/context budget exceeded')
        write(out/'runtime.json',dict(encoder=encoder.metadata,backend=backend.metadata,model=plan['model']))
        count=0
        with (out/'raw.jsonl').open('x') as raw,(out/'predictions.jsonl').open('x') as parsed:
            os.chmod(raw.name,0o600);os.chmod(parsed.name,0o600)
            for i,(row,encoded) in enumerate(zip(rows,entries)):
                result=backend.generate(encoded)
                record=dict(context=ctx,synthetic=plan['synthetic'],case_index=i,StudyInstanceUID=row['StudyInstanceUID'],
                    report_sha256=row['report_sha256'],prompt=candidate.make_prompt(key.split('-')[0],row['Report']),
                    rendered_prompt=encoded.rendered_prompt,input_ids=encoded.input_ids,generation=result)
                append(raw,record)  # Durable exact raw receipt before any parser can fail.
                response=candidate.validate_v3(result['raw_output'],row['Report'],result['generation_status'])
                append(parsed,dict(case_index=i,raw_record_sha256=digest(record),response=response))
                count+=1
                if result['generation_status']!='completed':break
        receipt.update(status='completed' if count==5 and result['generation_status']=='completed' else 'failed',
                       attempted=count,peak_allocated_gib=backend.peak_gib())
        checked_plan(prepared,expected_sha)
    except BaseException as error:
        receipt.update(status='failed',error_type=type(error).__name__)
        if not isinstance(error,Exception):raise
    finally:
        receipt.update(elapsed_seconds=time.monotonic()-started,finished_at=now(),
            artifacts={p.name:sha(p) for p in out.iterdir() if p.is_file()})
        write(out/'result.json',receipt)
    return receipt['status']=='completed'


def verify_run(prepared,key):
    root=Path(prepared)/'session';start=read(root/'start.json')
    plan,rows=checked_plan(prepared,start['plan_sha256']);out=root/key;receipt=read(out/'result.json')
    ctx=context(prepared,key)
    if receipt['status']!='completed' or receipt['synthetic']!=plan['synthetic'] or any(receipt.get(k)!=v for k,v in ctx.items()):
        raise ValueError('Run is incomplete, synthetic mismatch, or has wrong parent')
    required={'preflight.json','runtime.json','raw.jsonl','predictions.jsonl'} | (set() if plan['synthetic'] else {'cache_audit.json'})
    if set(receipt['artifacts'])!=required:raise ValueError('Incomplete run artifacts')
    for name,expected in receipt['artifacts'].items():
        if Path(name).name!=name or sha(out/name)!=expected:raise ValueError('Run artifact changed')
    if read(out/'runtime.json')['model']!=plan['model']:raise ValueError('Wrong effective model')
    if not plan['synthetic']:
        audit=read(out/'cache_audit.json')
        if not audit['complete'] or audit['revision']!=plan['model']['revision']:raise ValueError('Cache unverified')
    raw,parsed=lines(out/'raw.jsonl'),lines(out/'predictions.jsonl')
    if len(raw)!=5 or len(parsed)!=5 or receipt['attempted']!=5:raise ValueError('Missing or excess generations')
    if read(out/'preflight.json')['input_tokens']!=[r['generation']['input_tokens'] for r in raw]:
        raise ValueError('Preflight does not match generated inputs')
    for i,(record,pred,row) in enumerate(zip(raw,parsed,rows)):
        generation=record['generation']
        if (record['context']!=ctx or record['case_index']!=i or record['StudyInstanceUID']!=row['StudyInstanceUID']
            or record['report_sha256']!=row['report_sha256'] or record['synthetic']!=plan['synthetic']
            or record['prompt']!=candidate.make_prompt(key.split('-')[0],row['Report'])
            or generation['generation_status']!='completed' or pred['case_index']!=i
            or pred['raw_record_sha256']!=digest(record)
            or pred['response']!=candidate.validate_v3(generation['raw_output'],row['Report'],'completed')):
            raise ValueError('Output differs from exact input/provenance/reparse')
        from v2_runtime import generation_status
        elapsed=generation['runtime_seconds'];ids=generation['output_token_ids']
        if (not record['input_ids'] or any(type(n) is not int or n<0 for n in record['input_ids']+ids)
            or not math.isfinite(elapsed) or elapsed<0
            or generation_status(ids,[1,106],elapsed,300,4096)!='completed'
            or len(record['input_ids'])!=generation['input_tokens'] or
            len(generation['output_token_ids'])!=generation['output_tokens'] or
            generation['input_tokens']>plan['model']['max_input_tokens'] or
            generation['output_tokens']>4096 or generation['runtime_seconds']>=300):raise ValueError('Token/time budget mismatch')
    return receipt,raw,parsed


def execute_session(prepared,expected_sha,cache=None,execute=False,authorization=None,watchdog=None):
    plan,_=checked_plan(prepared,expected_sha)
    grant=None if plan['synthetic'] else guard(execute,prepared,expected_sha,authorization,watchdog)
    root=Path(prepared)/'session';root.mkdir(mode=0o700)
    if grant is not None:
        for source,name in ((authorization,'authorization.json'),(watchdog,'watchdog.json')):
            shutil.copyfile(source,root/name);(root/name).chmod(0o600)
    write(root/'start.json',dict(session_id=secrets.token_hex(16),parent_pid=os.getpid(),plan_sha256=expected_sha,
        synthetic=plan['synthetic'],run_order=ORDER,started_at=now(),
        authorization_sha256=None if grant is None else sha(authorization),
        watchdog_sha256=None if grant is None else sha(watchdog),provider_deadline=None if grant is None else grant['provider_deadline']))
    start=read(root/'start.json');results=[];began=time.monotonic();error_type=None
    try:
        for i,key in enumerate(ORDER):
            remaining=policy()['session_timeout_seconds']-(time.monotonic()-began)
            if grant is not None:
                guard(execute,prepared,expected_sha,authorization,watchdog)
                remaining=min(remaining,(parse_time(grant['provider_deadline'])-datetime.now(timezone.utc)).total_seconds()-policy()['reserved_copy_stop_seconds'])
            if remaining<=0:break
            nonce=secrets.token_hex(32)
            write(root/f'dispatch-{i}.json',dict(key=key,index=i,session_id=start['session_id'],
                start_sha256=sha(root/'start.json'),plan_sha256=expected_sha,nonce_sha256=digest(nonce)))
            r,w=os.pipe()
            with os.fdopen(w,'w') as stream:
                json.dump(dict(nonce=nonce,parent_pid=os.getpid(),key=key,plan_sha256=expected_sha),stream)
            command=[sys.executable,str(Path(__file__).resolve()),'_worker','--prepared',str(Path(prepared).resolve()),
                '--plan-sha256',expected_sha,'--key',key,'--worker-fd',str(r)]
            if not plan['synthetic']:command+=['--execute','--cache',str(Path(cache).resolve()),
                '--authorization',str(Path(authorization).resolve()),'--watchdog',str(Path(watchdog).resolve())]
            try:
                ok,reason=supervise(command,min(policy()['worker_timeout_seconds'],remaining),pass_fds=(r,))
            finally:os.close(r)
            if ok:
                try:verify_run(prepared,key)
                except Exception:ok=False;reason='receipt_verification_failed'
            results.append(dict(key=key,status='completed' if ok else 'failed',process_status=reason,
                receipt_sha256=sha(root/key/'result.json') if (root/key/'result.json').exists() else None))
            if not ok:break
    except BaseException as error:
        error_type=type(error).__name__
        if not isinstance(error,Exception):raise
    finally:
        elapsed=time.monotonic()-began
        complete=len(results)==4 and all(r['status']=='completed' for r in results) and elapsed<=policy()['session_timeout_seconds'] and error_type is None
        write(root/'result.json',dict(status='completed' if complete else 'failed',synthetic=plan['synthetic'],
            session_id=start['session_id'],plan_sha256=expected_sha,start_sha256=sha(root/'start.json'),
            results=results,error_type=error_type,elapsed_seconds=elapsed,finished_at=now(),
            model_calls=0 if plan['synthetic'] else None,provider_stopped=False,
            billing_stop_verified=False,expansion_allowed=False))
    return complete


def verify_session(prepared,allow_synthetic=False):
    root=Path(prepared)/'session';start=read(root/'start.json');final=read(root/'result.json')
    plan,rows=checked_plan(prepared,start['plan_sha256'])
    if plan['synthetic'] and not allow_synthetic:raise ValueError('Fabricated demo cannot be treated as model output')
    if not plan['synthetic']:
        if sha(root/'authorization.json')!=start['authorization_sha256'] or sha(root/'watchdog.json')!=start['watchdog_sha256']:
            raise ValueError('Original authorization/watchdog evidence changed')
        grant=read(root/'authorization.json');watch=read(root/'watchdog.json')
        if (grant['approved'] is not True or grant['plan_sha256']!=start['plan_sha256'] or
            grant['proposal_sha256']!=sha(ROOT/'configs/resource_proposal.json') or
            grant['provider_deadline']!=start['provider_deadline'] or watch['deadline']!=grant['provider_deadline'] or
            watch['pod_id']!=grant['pod_id'] or watch['script_sha256']!=sha(ROOT/'scripts/stop_watchdog.py')):
            raise ValueError('Execution provenance does not match the approved resource')
    if (final['status']!='completed' or final['synthetic']!=plan['synthetic'] or start['synthetic']!=plan['synthetic']
        or start['run_order']!=ORDER or final['session_id']!=start['session_id'] or
        final['plan_sha256']!=start['plan_sha256'] or final['start_sha256']!=sha(root/'start.json') or
        [r['key'] for r in final['results']]!=ORDER or not math.isfinite(final['elapsed_seconds']) or
        not 0<=final['elapsed_seconds']<=policy()['session_timeout_seconds']):
        raise ValueError('Incomplete or mismatched parent session')
    runs={}
    for result in final['results']:
        key=result['key']
        if result['status']!='completed' or result['receipt_sha256']!=sha(root/key/'result.json'):raise ValueError('Child result changed')
        runs[key]=verify_run(prepared,key)
    return plan,rows,runs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','demo','run','verify','_worker'])
    p.add_argument('--prepared',type=Path,required=True);p.add_argument('--candidate',type=Path)
    p.add_argument('--state',type=Path,default=Path(os.environ.get('RSNA_STATE_DIR',REPO/'state')))
    p.add_argument('--plan-sha256');p.add_argument('--cache',type=Path)
    p.add_argument('--execute',action='store_true');p.add_argument('--key',choices=ORDER)
    p.add_argument('--authorization',type=Path);p.add_argument('--watchdog',type=Path)
    p.add_argument('--worker-fd',type=int,help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.action=='prepare':print(json.dumps(prepare(a.prepared,a.candidate,a.state)))
    elif a.action=='demo':
        result=prepare(a.prepared,synthetic=True)
        ok=execute_session(a.prepared,result['plan_sha256'])
        print(json.dumps(dict(status='SYNTHETIC_DEMO_COMPLETE' if ok else 'FAILED',model_calls=0)))
        if not ok:raise SystemExit(1)
    elif a.action=='verify':
        verify_session(a.prepared);print('Complete measured session verified')
    elif a.action=='run':
        guard(a.execute,a.prepared,a.plan_sha256,a.authorization,a.watchdog)
        raise SystemExit(0 if execute_session(a.prepared,a.plan_sha256,a.cache,a.execute,a.authorization,a.watchdog) else 1)
    else:
        raise SystemExit(0 if run_once(a.prepared,a.plan_sha256,a.key,a.cache,a.worker_fd,a.execute,a.authorization,a.watchdog) else 1)


if __name__=='__main__':main()
