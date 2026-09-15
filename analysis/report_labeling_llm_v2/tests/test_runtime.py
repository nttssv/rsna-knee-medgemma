"""CPU fake-backend tests. No Transformers import, cache, download or real model."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
import sys
from types import SimpleNamespace

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
import core
import v2_runtime as rt
import run_smoke as runner


class Batch(dict):
    def to(self, device):
        assert device=='cuda:0'  # Fake transfer: tensors remain on CPU.
        return self


class FakeTokenizer:
    eos_token_id=1
    pad_token_id=0

    def __init__(self):
        self.calls=[]

    def __call__(self, text, **kwargs):
        self.calls.append((text,kwargs))
        return Batch(input_ids=torch.tensor([[7,8,9]]))

    def decode(self, ids, **kwargs):
        assert kwargs['clean_up_tokenization_spaces'] is False
        return ' synthetic output '


class FakeProcessor:
    chat_template='synthetic template only'

    def __init__(self):
        self.tokenizer=FakeTokenizer();self.calls=[]

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages,kwargs))
        if kwargs['tokenize']:
            return Batch(input_ids=torch.tensor([[7,8,9]]))
        return 'synthetic rendered prompt'


@pytest.mark.parametrize('model',['medgemma','qwen'])
def test_official_call_shapes_and_no_double_special_tokens(model):
    proc=FakeProcessor();tokenizer=proc.tokenizer
    encoded=rt.encode_prompt(proc,tokenizer,model,'original report instructions')
    assert encoded.input_ids==[7,8,9]
    assert all(call[1]['truncation'] is False and call[1]['add_special_tokens'] is False for call in tokenizer.calls)
    messages,kwargs=proc.calls[0]
    assert kwargs['add_generation_prompt'] and kwargs['tokenize'] is False
    if model=='medgemma':
        assert messages[0]['content']==[{'type':'text','text':'original report instructions'}]
        assert all('enable_thinking' not in kw for _,kw in proc.calls)
        assert proc.calls[1][1]['return_dict'] and proc.calls[1][1]['truncation'] is False
    else:
        assert messages[0]['content']=='original report instructions'
        assert kwargs['enable_thinking'] is False


def test_rendered_token_mismatch_rejected():
    proc=FakeProcessor()
    class WrongTokenizer(FakeTokenizer):
        def __call__(self,*args,**kwargs):return Batch(input_ids=torch.tensor([[99]]))
    with pytest.raises(ValueError,match='differs'):
        rt.encode_prompt(proc,WrongTokenizer(),'medgemma','synthetic')


@pytest.mark.parametrize('pad,token_pad,wanted',[(0,7,0),(None,0,0),(None,None,1),(151643,0,151643)])
def test_pad_zero_and_fallback_order(pad,token_pad,wanted):
    assert rt.resolve_stops([1,106],pad,token_pad,1)==([1,106],wanted)


@pytest.mark.parametrize('tokens,elapsed,wanted',[
    ([11,1],1,'completed'),([11,106],1,'completed'),([11,1],300,'timeout'),
    ([11,12],1,'generation_truncated'),([],1,'generation_truncated'),
    ([1,11,1],1,'generation_truncated'),([11,12,13,1],1,'generation_truncated')])
def test_eos_timeout_and_token_cap(tokens,elapsed,wanted):
    assert rt.generation_status(tokens,[1,106],elapsed,300,3)==wanted


def fake_backend(result=None,error=None):
    calls=[]
    cfg=core.config('medgemma')
    class Model:
        def generate(self,**kwargs):
            calls.append(kwargs)
            if error:raise error
            return torch.tensor([result if result is not None else [7,8,9,31,106]])
    class Encoder:
        def decode(self,ids):
            assert ids in ([31],[31,106])
            return 'FULL SPECIAL RAW' if ids==[31,106] else '{"fake":"text"}'
    fake_torch=SimpleNamespace(inference_mode=torch.inference_mode,
        cuda=SimpleNamespace(synchronize=lambda:None,empty_cache=lambda:None,
                             OutOfMemoryError=torch.cuda.OutOfMemoryError))
    b=SimpleNamespace(cfg=cfg,torch=fake_torch,model=Model(),encoder=Encoder(),
        context_limit=32768,eos=[1,106],gen=rt.generation_options(cfg,[1,106],0))
    return b,calls


def test_generate_slices_only_new_tokens_and_preserves_raw():
    b,calls=fake_backend();entry=rt.Encoded('rendered',Batch(input_ids=torch.tensor([[7,8,9]])),[7,8,9])
    ticks=iter([0.,2.])
    r=rt.generate_encoded(b,entry,clock=lambda:next(ticks))
    assert r['generation_status']=='completed' and r['output_token_ids']==[31,106]
    assert r['decoded_with_special_tokens']=='FULL SPECIAL RAW' and r['raw_output']=='{"fake":"text"}'
    assert len(calls)==1 and calls[0]['generation_config']['pad_token_id']==0
    assert calls[0]['generation_config']['do_sample'] is False


@pytest.mark.parametrize('error,status',[(torch.cuda.OutOfMemoryError('secret must not appear'),'oom'),(RuntimeError('secret must not appear'),'runtime_error')])
def test_generation_errors_are_not_repairs_or_secret_logs(error,status):
    b,calls=fake_backend(error=error);ticks=iter([0.,2.])
    r=rt.generate_encoded(b,rt.Encoded('x',Batch(input_ids=torch.tensor([[7,8,9]])),[7,8,9]),clock=lambda:next(ticks))
    assert r['generation_status']==status and len(calls)==1
    assert 'secret' not in json.dumps(r)
    assert all(row['label'] is None for row in core.validate_response(r['raw_output'],'',status)['rows'])


def test_wrong_prefix_and_context_overflow_never_accepted():
    b,calls=fake_backend(result=[0,8,9,31,106])
    r=rt.generate_encoded(b,rt.Encoded('x',Batch(input_ids=torch.tensor([[7,8,9]])),[7,8,9]))
    assert r['generation_status']=='runtime_error'
    b.context_limit=2;calls.clear()
    assert rt.generate_encoded(b,rt.Encoded('x',None,[7,8,9]))['generation_status']=='context_overflow'
    assert not calls


def prepared_fixture(tmp_path,monkeypatch):
    prepared=tmp_path/'prepared';prepared.mkdir()
    rows=[dict(StudyInstanceUID=f'synthetic-{i}',Report='Synthetic: ACL is intact.',split='development',language='Synthetic',report_sha256=rt.text_sha('Synthetic: ACL is intact.')) for i in range(5)]
    (prepared/'inputs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    for m in ('medgemma','qwen'):
        (prepared/f'{m}_prompts.jsonl').write_text(''.join(json.dumps(dict(case_index=i,prompt=core.prompt_for(m,r['Report'])))+'\n' for i,r in enumerate(rows)))
    original=runner.config
    runtime=copy.deepcopy(original('runtime'));runtime['fixed_inputs_sha256']=core.sha(prepared/'inputs.jsonl')
    monkeypatch.setattr(runner,'config',lambda name:runtime if name=='runtime' else original(name))
    m=dict(candidate_code_sha256=core.code_hashes(),source_sha256=core.config('input_anchors')['source_sha256'],
        split_sha256=core.config('input_anchors')['split_sha256'],files={p.name:core.sha(p) for p in prepared.iterdir()})
    (prepared/'manifest.json').write_text(json.dumps(m))
    return prepared


class EngineEncoder:
    metadata={'synthetic':True}
    def encode(self,prompt):return rt.Encoded('SYNTHETIC RENDER: '+prompt,None,[7,8,9])


class EngineBackend:
    context_limit=32768
    metadata={'synthetic':True}
    def __init__(self,status='completed'):self.calls=0;self.status=status
    def peak_gib(self):return None
    def generate(self,entry):
        self.calls+=1
        raw=json.dumps({c:dict(label='not_mentioned',evidence_text='',confidence=0) for c in core.LABELS})
        return dict(raw_output=raw,decoded_with_special_tokens=raw+'<SYNTHETIC_EOS>',output_token_ids=[5,1],input_tokens=3,output_tokens=2,runtime_seconds=.1,generation_status=self.status)


def test_plan_is_fixed_offline_and_changed_input_rejected(tmp_path,monkeypatch):
    p=prepared_fixture(tmp_path,monkeypatch);plan=runner.make_plan(p)
    assert plan['model_calls']==0 and plan['tokenizer_preflight']=='NOT_RUN' and plan['run_order']==runner.ORDER
    assert len(plan['cases'])==5 and plan['maximum_primary_generations']==20
    target=p/'execution_plan.json';runner.write_json(target,plan)
    assert runner.check_plan(p,target,core.sha(target))==plan
    with pytest.raises(ValueError,match='SHA'):runner.check_plan(p,target,'bad')
    (p/'inputs.jsonl').write_text('tampered')
    with pytest.raises(ValueError):runner.make_plan(p)


def test_fake_engine_only_five_raw_records_before_parser_and_no_overwrite(tmp_path,monkeypatch):
    p=prepared_fixture(tmp_path,monkeypatch);out=p/'synthetic-run';backend=EngineBackend()
    original=runner.validate_response
    def checked_parser(*args):
        assert len((out/'raw_generations.jsonl').read_text().splitlines())==backend.calls
        return original(*args)
    monkeypatch.setattr(runner,'validate_response',checked_parser)
    assert runner.run_once(p,runner.make_plan(p),'medgemma-1',out,lambda _:EngineEncoder(),lambda _:backend,synthetic=True)
    manifest=json.loads((out/'run_manifest.json').read_text())
    records=[json.loads(s) for s in (out/'predictions.jsonl').read_text().splitlines()]
    assert len(records)==backend.calls==5 and manifest['synthetic'] is True
    assert manifest['predictions_sha256']==core.sha(out/'predictions.jsonl')
    assert all(r['synthetic'] and r['response']['rows'][0]['label']=='not_mentioned' for r in records)
    with pytest.raises(FileExistsError):runner.run_once(p,runner.make_plan(p),'medgemma-1',out,lambda _:EngineEncoder(),lambda _:backend,synthetic=True)


@pytest.mark.parametrize('status',['timeout','oom','generation_truncated','runtime_error'])
def test_failed_generation_stops_without_repair(tmp_path,monkeypatch,status):
    p=prepared_fixture(tmp_path,monkeypatch);out=p/'synthetic-run';backend=EngineBackend(status)
    assert not runner.run_once(p,runner.make_plan(p),'qwen-1',out,lambda _:EngineEncoder(),lambda _:backend,synthetic=True)
    assert backend.calls==1
    records=[json.loads(s) for s in (out/'predictions.jsonl').read_text().splitlines()]
    assert len(records)==1 and all(r['label'] is None for r in records[0]['response']['rows'])


def test_context_preflight_blocks_model_loading(tmp_path,monkeypatch):
    p=prepared_fixture(tmp_path,monkeypatch)
    class LongEncoder(EngineEncoder):
        def encode(self,prompt):return rt.Encoded('synthetic',None,list(range(8193)))
    def forbidden(_):pytest.fail('Weights must not load after token overflow')
    assert not runner.run_once(p,runner.make_plan(p),'qwen-1',p/'overflow',lambda _:LongEncoder(),forbidden,synthetic=True)


def test_supervisor_kills_timed_out_worker_and_does_not_claim_provider_stop(monkeypatch):
    signals=[]
    class Process:
        pid=12345
        def wait(self,timeout=None):
            if timeout is not None:raise subprocess.TimeoutExpired('synthetic',timeout)
            return -9
    def launch(command,**kwargs):
        assert kwargs==dict(start_new_session=True)
        return Process()
    monkeypatch.setattr(runner.os,'killpg',lambda pid,sig:signals.append((pid,sig)))
    assert runner.supervise(['synthetic'],4,launcher=launch)==(False,'hard_timeout')
    assert signals==[(12345,runner.signal.SIGKILL)]


def test_real_cpu_watchdog_reaps_worker(tmp_path):
    marker=tmp_path/'pid'
    code='import os,time,pathlib; pathlib.Path('+repr(str(marker))+').write_text(str(os.getpid())); time.sleep(10)'
    assert runner.supervise([sys.executable,'-c',code],.5)==(False,'hard_timeout')
    assert marker.exists()
    with pytest.raises(ProcessLookupError):os.kill(int(marker.read_text()),0)


def test_locked_cli_fails_before_transformers_or_any_cache_access(tmp_path):
    result=subprocess.run([sys.executable,str(SCRIPTS/'run_smoke.py'),'run','--prepared',str(tmp_path),
        '--plan',str(tmp_path/'plan.json'),'--execute'],capture_output=True,text=True)
    assert result.returncode!=0 and 'GPU execution is locked' in result.stderr
    assert not list(tmp_path.iterdir())
    assert 'transformers' not in sys.modules


def test_hf_encoder_pins_revision_and_offline_access(monkeypatch,tmp_path):
    proc=FakeProcessor();calls=[]
    snapshot=tmp_path/'models--google--medgemma-1.5-4b-it'/'snapshots'/core.config('medgemma')['revision']
    snapshot.mkdir(parents=True);(snapshot/'config.json').write_text('{}')
    class Loader:
        @staticmethod
        def from_pretrained(model,**kwargs):calls.append((model,kwargs));return proc
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoProcessor=Loader,AutoTokenizer=Loader))
    original=rt.config
    policy=copy.deepcopy(original('runtime'));policy['models']['medgemma']['chat_template_sha256']=rt.text_sha(proc.chat_template)
    monkeypatch.setattr(rt,'config',lambda name:policy if name=='runtime' else original(name))
    encoder=rt.HFEncoder('medgemma',tmp_path)
    assert encoder.metadata['thinking_control']=='no_supported_switch_claimed'
    assert calls[0][1]['local_files_only'] is True and calls[0][1]['trust_remote_code'] is False
    assert calls[0][1]['revision']==core.config('medgemma')['revision']
    policy['models']['medgemma']['chat_template_sha256']='changed'
    with pytest.raises(ValueError,match='template'):rt.HFEncoder('medgemma',tmp_path)


def test_hf_backend_constructor_pins_offline_weights_and_preserves_pad_zero(monkeypatch,tmp_path):
    cfg=core.config('medgemma');loads=[];seeds=[]
    snapshot=tmp_path/'models--google--medgemma-1.5-4b-it'/'snapshots'/cfg['revision']
    snapshot.mkdir(parents=True);(snapshot/'config.json').write_text('{}')
    model=SimpleNamespace(config=SimpleNamespace(_commit_hash=cfg['revision'],text_config=SimpleNamespace(max_position_embeddings=32768)),
                          generation_config=SimpleNamespace(eos_token_id=[1,106],pad_token_id=0),eval=lambda:None)
    class Loader:
        @staticmethod
        def from_pretrained(*args,**kwargs):loads.append((args,kwargs));return model
    class Gen:
        def __init__(self,**kwargs):self.kwargs=kwargs
        def to_dict(self):return self.kwargs
    fake_cuda=SimpleNamespace(is_available=lambda:True,is_bf16_supported=lambda:True,
        mem_get_info=lambda:(80*2**30,80*2**30),reset_peak_memory_stats=lambda:None,
        get_device_name=lambda _: 'SYNTHETIC GPU',max_memory_allocated=lambda:0)
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=fake_cuda,bfloat16='BF16',version=SimpleNamespace(cuda='SYNTHETIC')))
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoModelForImageTextToText=Loader,GenerationConfig=Gen,set_seed=seeds.append))
    monkeypatch.setattr(rt,'check_versions',lambda:{'synthetic':True})
    encoder=SimpleNamespace(cfg=cfg,model_key='medgemma',tokenizer=FakeTokenizer(),metadata={'synthetic':True})
    b=rt.HFBackend(encoder,tmp_path)
    assert b.pad==0 and b.eos==[1,106] and seeds==[cfg['seed']]
    assert loads[0][1]['revision']==cfg['revision'] and loads[0][1]['local_files_only'] is True
    assert loads[0][1]['trust_remote_code'] is False and loads[0][1]['device_map']=={'':'cuda:0'}
    assert b.options['num_return_sequences']==1 and b.options['max_new_tokens']==2048
    model.config._commit_hash='changed'
    with pytest.raises(ValueError,match='revision'):rt.HFBackend(encoder,tmp_path)


def test_partial_parse_failure_keeps_raw_and_failed_manifest(tmp_path,monkeypatch):
    p=prepared_fixture(tmp_path,monkeypatch);out=p/'parse-crash';backend=EngineBackend()
    def fail(*args):raise ValueError('fake parser interruption')
    monkeypatch.setattr(runner,'validate_response',fail)
    assert not runner.run_once(p,runner.make_plan(p),'qwen-1',out,lambda _:EngineEncoder(),lambda _:backend,synthetic=True)
    assert len((out/'raw_generations.jsonl').read_text().splitlines())==1
    assert json.loads((out/'run_manifest.json').read_text())['status']=='failed'
    assert (out/'predictions.jsonl').read_text()==''


def test_decoder_preserves_internal_special_tokens(monkeypatch,tmp_path):
    calls=[]
    encoder=rt.HFEncoder.__new__(rt.HFEncoder)
    class Tok:
        def decode(self,ids,**kwargs):calls.append((ids,kwargs));return '<unexpected>'+str(ids)
    encoder.tokenizer=Tok()
    assert encoder.decode([5,6]).startswith('<unexpected>')
    assert calls==[([5,6],dict(skip_special_tokens=False,clean_up_tokenization_spaces=False))]


def test_empty_cache_fails_before_transformers_import(tmp_path,monkeypatch):
    monkeypatch.delitem(sys.modules,'transformers',raising=False)
    with pytest.raises(ValueError,match='pinned snapshot'):rt.HFEncoder('medgemma',tmp_path)
    assert 'transformers' not in sys.modules


def test_explicit_offline_environment_never_uses_default_cache(tmp_path,monkeypatch):
    for key in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_HOME','HF_HUB_CACHE','TRANSFORMERS_CACHE','HF_HUB_DISABLE_IMPLICIT_TOKEN'):
        monkeypatch.setenv(key,'wrong')
    snapshot=tmp_path/'models--Qwen--Qwen3-14B'/'snapshots'/core.config('qwen')['revision']
    snapshot.mkdir(parents=True);(snapshot/'config.json').write_text('{}')
    assert rt.offline_environment(tmp_path,'qwen')==tmp_path
    assert os.environ['HF_HUB_OFFLINE']==os.environ['TRANSFORMERS_OFFLINE']=='1'
    assert all(os.environ[k]==str(tmp_path) for k in ('HF_HOME','HF_HUB_CACHE','TRANSFORMERS_CACHE'))


def test_watchdog_stops_descendant_process(tmp_path):
    pidfile=tmp_path/'descendant'
    grandchild='import time; time.sleep(20)'
    child='import subprocess,sys,pathlib,time; p=subprocess.Popen([sys.executable,"-c",'+repr(grandchild)+']); pathlib.Path('+repr(str(pidfile))+').write_text(str(p.pid)); time.sleep(20)'
    assert runner.supervise([sys.executable,'-c',child],.5)==(False,'hard_timeout')
    assert pidfile.exists()
    pid=pidfile.read_text()
    deadline=time.monotonic()+2
    while True:
        status=subprocess.run(['ps','-p',pid,'-o','stat='],capture_output=True,text=True).stdout.strip()
        if not status or status.startswith('Z'):break  # Dead, possibly awaiting init reaping.
        assert time.monotonic()<deadline,'Descendant remained live after process-group kill'
        time.sleep(.02)
