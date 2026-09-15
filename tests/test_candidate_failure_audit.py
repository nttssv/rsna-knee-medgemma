"""Synthetic counterexamples for the offline receipt and token audit."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

P=Path(__file__).resolve().parents[1]/'analysis/candidate_failure_diagnostic_v1/audit.py'
spec=importlib.util.spec_from_file_location('candidate_failure_audit',P)
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


class Encoder:
    def encode(self,prompt):return SimpleNamespace(rendered_prompt='rendered',input_ids=[2,3])
    def decode(self,ids):return { (5,):'answer',(5,1):'answer<EOS>' }[tuple(ids)]


def record():
    return dict(prompt='fixture',rendered_prompt='rendered',input_ids=[2,3],
        generation=dict(output_token_ids=[5,1],input_tokens=2,output_tokens=2,
            raw_output='answer',decoded_with_special_tokens='answer<EOS>',
            runtime_seconds=1.,generation_status='completed'))


def test_exact_eos_handling():
    assert a.token_roundtrip(Encoder(),{},record())


@pytest.mark.parametrize('field,value', [('input_ids',[3,2]),('rendered_prompt','changed')])
def test_input_corruption_rejected(field,value):
    r=record();r[field]=value
    with pytest.raises(ValueError,match='Input render'):a.token_roundtrip(Encoder(),{},r)


def test_output_corruption_rejected():
    r=record();r['generation']['raw_output']='answer<EOS>'
    with pytest.raises(ValueError,match='Output decode'):a.token_roundtrip(Encoder(),{},r)


def test_wrong_status_rejected():
    r=record();r['generation']['generation_status']='generation_truncated'
    with pytest.raises(ValueError,match='Generation status'):a.token_roundtrip(Encoder(),{},r)


def test_thought_and_answer_are_not_conflated():
    assert a.token_layout([90,7,7,91,8,1],90,91)['thought_tokens']==2
    assert a.token_layout([90,7,7,91,8,1],90,91)['answer_surface_tokens']==1
    assert a.token_layout([90,7,7],90,91)['layout']=='unfinished_thought'
    x=a.token_layout([8]*64,90,91)
    assert x['thought_tokens']==0 and x['answer_surface_tokens']==64
    assert x['max_identical_16_token_window_occurrences']==49


def test_string_states_are_not_rescued():
    result=a.json_shape(dict(normalized_output=json.dumps({c:'negative' for c in a.rt.core.LABELS})))
    assert result['string_state_values']==12 and result['converted_to_labels'] is False


def test_partial_list_repetition_is_descriptive_only():
    raw='```json\n[\n'+('  {"evidence_text":"fictional passage"},\n'*4)
    x=a.surface_structure(raw)
    assert x==dict(starts_with_fenced_array=True,object_shaped_lines=4,distinct_object_shaped_lines=1,most_frequent_exact_object_line_count=4,labels_rescued=0)


@pytest.fixture
def receipt_fixture(tmp_path,monkeypatch):
    rt=a.rt;root=tmp_path/'session';root.mkdir()
    def write(p,v):p.write_text(json.dumps(v))
    deadline='2026-09-15T09:30:00+00:00'
    grant=dict(plan_sha256=a.PLAN_SHA,proposal_sha256=rt.sha(a.EXEC/'configs/resource_proposal.json'),approved=True,pod_id='fictional',provider_deadline=deadline)
    watch=dict(pod_id='fictional',deadline=deadline,status='armed',cli_syntax_and_read_access_verified=True,script_sha256=rt.sha(a.EXEC/'scripts/stop_watchdog.py'))
    write(root/'authorization.json',grant);write(root/'watchdog.json',watch)
    start=dict(session_id='fixture',run_order=rt.ORDER,synthetic=False,plan_sha256=a.PLAN_SHA,provider_deadline=deadline,started_at='2026-09-15T08:50:00+00:00',**{n+'_sha256':rt.sha(root/(n+'.json')) for n in ['authorization','watchdog']})
    write(root/'start.json',start)
    parent=dict(status='failed',synthetic=False,error_type=None,session_id='fixture',start_sha256=rt.sha(root/'start.json'),plan_sha256=a.PLAN_SHA,finished_at='2026-09-15T09:04:00+00:00',results=[])
    model={'fixture':True};plan=dict(model=model)
    manifest=rt.read(a.REPO/'analysis/report_labeling_llm_v2/configs/weight_manifest.json')['medgemma']
    for i,(key,n) in enumerate([('control-1',5),('candidate-1',4)]):
        write(root/f'dispatch-{i}.json',dict(key=key,index=i,session_id='fixture',start_sha256=rt.sha(root/'start.json'),plan_sha256=a.PLAN_SHA))
        out=root/key;out.mkdir()
        write(out/'cache_audit.json',dict(complete=True,model='medgemma',revision=manifest['revision'],files=[dict(file=f['file'],bytes=f['bytes'],expected_bytes=f['bytes'],digest=f['digest'],verified=True) for f in manifest['files']]))
        encoder=dict(fixture=True)
        write(out/'preflight.json',dict(encoder=encoder,input_tokens=[2]*5))
        backend=dict(**encoder,effective_v3_model_config=model,package_versions=rt.core.config('runtime')['required_versions'],gpu_name='NVIDIA RTX 6000 Ada Generation',nvidia_driver_version='570.124.06',cuda_version='12.8',dtype='bfloat16',attention='sdpa',seed=20260914,effective_generation_config=dict(max_new_tokens=4096,max_time=300,do_sample=False,num_beams=1,eos_token_id=[1,106],pad_token_id=0))
        write(out/'runtime.json',dict(model=model,encoder=encoder,backend=backend))
        statuses=['completed']*n
        if n==4:statuses[-1]='generation_truncated'
        (out/'raw.jsonl').write_text('\n'.join(json.dumps(dict(generation=dict(generation_status=s,input_tokens=2,runtime_seconds=1))) for s in statuses))
        (out/'predictions.jsonl').write_text('{}\n'*n)
        status='completed' if n==5 else 'failed'
        result=dict(**rt.context(tmp_path,key),status=status,attempted=n,synthetic=False,started_at='2026-09-15T08:51:00+00:00',finished_at='2026-09-15T09:03:00+00:00',elapsed_seconds=720,artifacts={p.name:rt.sha(p) for p in out.iterdir()})
        write(out/'result.json',result)
        parent['results'].append(dict(key=key,status=status,process_status='completed' if n==5 else 'child_failed',receipt_sha256=rt.sha(out/'result.json')))
    monkeypatch.setattr(rt,'verify_run',lambda *args:None)
    return tmp_path,plan,parent


def test_both_saved_child_receipts_are_audited(receipt_fixture):
    assert [r['recorded'] for r in a.audit_receipts(*receipt_fixture)]==[5,4]


def test_added_later_dispatch_rejected(receipt_fixture):
    p,plan,parent=receipt_fixture;(p/'session/dispatch-2.json').write_text('{}')
    with pytest.raises(ValueError,match='later dispatch'):a.audit_receipts(p,plan,parent)


def test_modified_artifact_rejected(receipt_fixture):
    p,plan,parent=receipt_fixture;(p/'session/candidate-1/runtime.json').write_text('{}')
    with pytest.raises(ValueError,match='artifact binding'):a.audit_receipts(p,plan,parent)


def test_rebound_wrong_failed_count_rejected(receipt_fixture):
    p,plan,parent=receipt_fixture;file=p/'session/candidate-1/result.json';v=json.loads(file.read_text());v['attempted']=5;file.write_text(json.dumps(v));parent['results'][1]['receipt_sha256']=a.rt.sha(file)
    with pytest.raises(ValueError,match='Child receipt/context'):a.audit_receipts(p,plan,parent)
