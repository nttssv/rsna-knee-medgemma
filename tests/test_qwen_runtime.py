"""CPU integration/adversarial checks; all reports and responses here are fabricated."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

SCRIPTS=Path(__file__).resolve().parents[1]/'analysis/qwen_runtime_v1/scripts'
sys.path.insert(0,str(SCRIPTS))
import runtime_qwen as rt
import review_qwen as review


def replace(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


@pytest.fixture(scope='module')
def completed(tmp_path_factory):
    root=tmp_path_factory.mktemp('qwen-demo')/'prepared'
    result=rt.prepare(root,synthetic=True)
    assert rt.execute_session(root,result['plan_sha256'])
    rt.verify_session(root,allow_synthetic=True)
    return root


@pytest.fixture
def session(completed,tmp_path):
    dest=tmp_path/'prepared';shutil.copytree(completed,dest)
    return dest


@pytest.fixture
def bundle(session,tmp_path):
    output=tmp_path/'review';review.make_bundle(session,output,True)
    return output


def review_rows(bundle):
    rows=rt.read(bundle/'blinded/review_template.json')
    for row in rows:row.update(reviewer='Synthetic software fixture reviewer',reviewed_at='2026-09-15T10:00:00+00:00',categories=['no_issue_identified'])
    return rows


def refresh_run(session,key):
    root=session/'session';out=root/key
    result=rt.read(out/'result.json')
    result['artifacts']={name:rt.sha(out/name) for name in result['artifacts']};replace(out/'result.json',result)
    final=rt.read(root/'result.json')
    for r in final['results']:
        if r['key']==key:r['receipt_sha256']=rt.sha(out/'result.json')
    replace(root/'result.json',final)


def worker_fixture(tmp_path):
    prepared=tmp_path/'prepared';result=rt.prepare(prepared,synthetic=True)
    root=prepared/'session';root.mkdir()
    rt.write(root/'start.json',dict(session_id='synthetic-test',parent_pid=os.getppid(),plan_sha256=result['plan_sha256'],synthetic=True,run_order=rt.ORDER))
    rt.write(root/'dispatch-0.json',dict(key=rt.ORDER[0],index=0,session_id='synthetic-test',start_sha256=rt.sha(root/'start.json'),plan_sha256=result['plan_sha256'],nonce_sha256=rt.digest('fixture')))
    r,w=os.pipe()
    with os.fdopen(w,'w') as f:json.dump(dict(nonce='fixture',parent_pid=os.getppid(),key=rt.ORDER[0],plan_sha256=result['plan_sha256']),f)
    return prepared,result['plan_sha256'],r


def test_history_and_model_recipe():
    assert rt.verify_history()==203
    cfg=rt.model_config()
    assert cfg['max_new_tokens']==2048 and rt.core.config('qwen')['max_new_tokens']==2048
    assert cfg['model_id']=='Qwen/Qwen3-14B'
    assert cfg['revision']==rt.candidate.read(rt.QWEN/'configs/experiment.json')['revision']
    assert rt.ORDER==['control-1','candidate-1','candidate-2','control-2']
    assert not rt.policy()['execution_enabled']


@pytest.mark.parametrize('execute',[False,True])
def test_guard_cannot_unlock(execute,monkeypatch):
    with pytest.raises(PermissionError):rt.guard(execute)
    p=rt.policy();p.update(execution_enabled=True,resource_authorization='not sufficient')
    monkeypatch.setattr(rt,'policy',lambda:p)
    with pytest.raises(PermissionError):rt.guard(execute)


def test_cli_run_refuses_before_inputs(tmp_path):
    result=subprocess.run([sys.executable,str(SCRIPTS/'runtime_qwen.py'),'run','--prepared',str(tmp_path/'absent'),'--execute'],capture_output=True,text=True)
    assert result.returncode!=0 and 'GPU execution disabled' in result.stderr
    assert not (tmp_path/'absent').exists()


def test_preserve_existing_and_private_boundary(completed):
    with pytest.raises(FileExistsError):rt.prepare(completed,synthetic=True)
    with pytest.raises(ValueError):rt.private(rt.REPO/'analysis/leak')
    with pytest.raises(ValueError):rt.prepare(completed.parent/'unused',source=completed,synthetic=True)


@pytest.mark.parametrize('change',['studies','maximum_generations','run_order','model','candidate_sha256','runtime_sha256'])
def test_stale_plan_rejected(session,change):
    plan=rt.read(session/'plan.json')
    plan[change]=None;replace(session/'plan.json',plan)
    with pytest.raises(ValueError):rt.checked_plan(session,rt.sha(session/'plan.json'))


def test_demo_cannot_consume_real_or_extra_report(session):
    rows=rt.lines(session/'inputs.jsonl');rows[0]['Report']='An unauthorized report'
    rows[0]['report_sha256']=rt.hashlib.sha256(rows[0]['Report'].encode()).hexdigest()
    (session/'inputs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    plan=rt.read(session/'plan.json');plan['files']['inputs.jsonl']=rt.sha(session/'inputs.jsonl');replace(session/'plan.json',plan)
    with pytest.raises(ValueError):rt.checked_plan(session,rt.sha(session/'plan.json'))


def test_complete_demo_bound_and_not_measured(completed):
    with pytest.raises(ValueError,match='Fabricated'):rt.verify_session(completed)
    plan,rows,runs=rt.verify_session(completed,True)
    assert plan['synthetic'] and len(rows)==5 and list(runs)==rt.ORDER
    assert sum(len(v[1]) for v in runs.values())==20
    assert rt.read(completed/'session/result.json')['model_calls']==0
    summary=review.technical_summary(completed,True)
    assert summary['semantic_accuracy'] is None and not summary['expansion_allowed']
    assert sum(r['binary_decisions'] for r in summary['condition_results'])==0
    assert all(not r['condition_differences'] for r in summary['repeat_consistency'])


@pytest.mark.parametrize('change',['missing_run','wrong_parent','wrong_input','wrong_prompt','wrong_parser','missing_cell','token_cap','output_order','raw_changed','wrong_dispatch','wrong_synthetic'])
def test_session_rejects_mismatched_receipts(session,change):
    root=session/'session';key=rt.ORDER[0];out=root/key
    if change=='missing_run':shutil.rmtree(root/rt.ORDER[-1])
    elif change=='wrong_parent':
        result=rt.read(out/'result.json');result['session_id']='other';replace(out/'result.json',result);refresh_run(session,key)
    elif change=='wrong_dispatch':
        dispatch=rt.read(root/'dispatch-0.json');dispatch['key']=rt.ORDER[1];replace(root/'dispatch-0.json',dispatch)
    elif change in ('wrong_parser','missing_cell'):
        rows=rt.lines(out/'predictions.jsonl')
        if change=='missing_cell':rows[0]['response']['rows'].pop()
        else:rows[0]['response']['rows'][0]['label']='negative'
        (out/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));refresh_run(session,key)
    else:
        rows=rt.lines(out/'raw.jsonl')
        if change=='wrong_input':rows[0]['StudyInstanceUID']='other'
        elif change=='wrong_prompt':rows[0]['prompt']='new prompt'
        elif change=='token_cap':rows[0]['generation']['output_tokens']=4097
        elif change=='output_order':rows.reverse()
        elif change=='wrong_synthetic':rows[0]['synthetic']=False
        else:rows[0]['generation']['raw_output']+='changed'
        (out/'raw.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));refresh_run(session,key)
    with pytest.raises((ValueError,FileNotFoundError)):rt.verify_session(session,True)


def test_worker_needs_pipe(tmp_path):
    with pytest.raises(PermissionError):rt.verify_parent(tmp_path,rt.ORDER[0],None)
    path=tmp_path/'regular';path.write_text('{}')
    with path.open() as f:
        with pytest.raises(PermissionError):rt.verify_parent(tmp_path,rt.ORDER[0],f.fileno())


def test_worker_rejects_wrong_nonce(tmp_path):
    prepared,expected,fd=worker_fixture(tmp_path)
    p=prepared/'session/dispatch-0.json';d=rt.read(p);d['nonce_sha256']=rt.digest('wrong');replace(p,d)
    with pytest.raises(PermissionError):rt.run_once(prepared,expected,rt.ORDER[0],worker_fd=fd)


def test_truncation_stops_without_retry(tmp_path,monkeypatch):
    prepared,expected,fd=worker_fixture(tmp_path)
    class Truncated(rt.SyntheticBackend):
        calls=0
        def generate(self,encoded):
            self.calls+=1;result=super().generate(encoded)
            if self.calls==3:result['generation_status']='generation_truncated'
            return result
    monkeypatch.setattr(rt,'SyntheticBackend',Truncated)
    assert not rt.run_once(prepared,expected,rt.ORDER[0],worker_fd=fd)
    out=prepared/'session'/rt.ORDER[0]
    assert len(rt.lines(out/'raw.jsonl'))==3
    assert all(r['label'] is None for r in rt.lines(out/'predictions.jsonl')[-1]['response']['rows'])
    assert rt.read(out/'result.json')['status']=='failed'


def test_raw_durable_before_parser_exception(tmp_path,monkeypatch):
    prepared,expected,fd=worker_fixture(tmp_path)
    def broken(*args):raise RuntimeError('synthetic parser crash')
    monkeypatch.setattr(rt,'validate_primary',broken)
    assert not rt.run_once(prepared,expected,rt.ORDER[0],worker_fd=fd)
    out=prepared/'session'/rt.ORDER[0]
    assert len(rt.lines(out/'raw.jsonl'))==1 and not rt.lines(out/'predictions.jsonl')
    assert rt.read(out/'result.json')['error_type']=='RuntimeError'


def test_parent_failure_and_interrupt_preserve_receipt(tmp_path,monkeypatch):
    for interrupt in (False,True):
        prepared=tmp_path/str(interrupt);result=rt.prepare(prepared,synthetic=True)
        def fail(*args,**kwargs):
            if interrupt:raise KeyboardInterrupt()
            return False,'timeout'
        monkeypatch.setattr(rt,'supervise',fail)
        if interrupt:
            with pytest.raises(KeyboardInterrupt):rt.execute_session(prepared,result['plan_sha256'])
        else:assert not rt.execute_session(prepared,result['plan_sha256'])
        receipt=rt.read(prepared/'session/result.json')
        assert receipt['status']=='failed' and not receipt['provider_stopped']
        assert len(list((prepared/'session').glob('dispatch-*.json')))==1
        with pytest.raises(ValueError):rt.verify_session(prepared,True)


def test_blind_bundle_omits_arm_gold_identifiers(session,bundle):
    plan,inputs,runs,mapping,cases=review.verify_bundle(session,bundle,True)
    payload=''.join(p.read_text() for p in (bundle/'blinded').iterdir())
    for forbidden in rt.ORDER+['StudyInstanceUID','raw_record_sha256','reference_discordance','<unused94>']+[r['StudyInstanceUID'] for r in inputs]:
        assert forbidden not in payload
    assert len(cases)==20 and len({m['review_id'] for m in mapping['mapping']})==20
    assert all(not row['categories'] for row in rt.read(bundle/'blinded/review_template.json'))
    assert 'SYNTHETIC SOFTWARE DEMO' in payload
    with pytest.raises(ValueError):review.verify_bundle(session,bundle)


def test_blank_review_does_not_unlock_reference(session,bundle,tmp_path):
    with pytest.raises(ValueError):review.finalize(session,bundle,bundle/'blinded/review_template.json',tmp_path/'final',True)
    assert not (tmp_path/'final').exists()


@pytest.mark.parametrize('change',['duplicate','missing','hash','reviewer','time','unknown_category','empty_category','duplicate_category','exclusive','reference','no_notes','swap_id','extra_field'])
def test_bad_review_rejected(session,bundle,tmp_path,change):
    rows=review_rows(bundle)
    if change=='duplicate':rows[-1]=copy.deepcopy(rows[0])
    elif change=='missing':rows.pop()
    elif change=='hash':rows[0]['output_sha256']='stale'
    elif change=='reviewer':rows[0]['reviewer']=' '
    elif change=='time':rows[0]['reviewed_at']='2026-09-15T10:00:00'
    elif change=='unknown_category':rows[0]['categories']=['correct']
    elif change=='empty_category':rows[0]['categories']=[]
    elif change=='duplicate_category':rows[0]['categories']*=2
    elif change=='exclusive':rows[0]['categories']+=['polarity_contradiction']
    elif change=='reference':rows[0]['categories']=['reference_discordance']
    elif change=='no_notes':rows[0]['categories']=['insufficient_information']
    elif change=='swap_id':rows[0]['review_id']='other session'
    else:rows[0]['arm']='control'
    path=tmp_path/'review.json';rt.write(path,rows)
    with pytest.raises(ValueError):review.validate_review(session,bundle,path,True)


def test_completed_synthetic_review_not_clinical_approval(session,bundle,tmp_path):
    path=tmp_path/'review.json';rt.write(path,review_rows(bundle))
    result=review.finalize(session,bundle,path,tmp_path/'final',True)
    assert result['reviewed_cells']==240 and not result['expansion_allowed']
    report=rt.read(tmp_path/'final/review_results.json')
    assert not report['reviewer_credentials_independently_verified']
    assert sum(r.get('reviewer_no_issue_binary_yield',0) for r in report['condition_results'])==0
    assert all(r['organizer'] is None for r in rt.read(tmp_path/'final/organizer_comparison.json')['rows'])


def test_bundle_mapping_tamper_rejected(session,bundle):
    path=bundle/'operator/mapping.json';mapping=rt.read(path)
    mapping['mapping'][0]['case_index']=4;replace(path,mapping)
    with pytest.raises(ValueError):review.verify_bundle(session,bundle,True)


def test_blinded_html_changed_rejected(session,bundle):
    with (bundle/'blinded/index.html').open('a') as f:f.write('changed')
    with pytest.raises(ValueError):review.verify_bundle(session,bundle,True)


@pytest.fixture
def failed_bundle(session,tmp_path):
    """Fabricated completed generation with a parse failure; never measured output."""
    key=rt.ORDER[0];out=session/'session'/key
    raw=rt.lines(out/'raw.jsonl');raw[0]['generation'].update(raw_output='synthetic invalid JSON',decoded_with_special_tokens='synthetic invalid JSON')
    (out/'raw.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in raw))
    parsed=rt.lines(out/'predictions.jsonl')
    parsed[0]=dict(case_index=0,raw_record_sha256=rt.digest(raw[0]),response=rt.validate_primary(raw[0]['generation']['raw_output'],'Synthetic software fixture 0.'),secondary_response=rt.validate_secondary(raw[0]['generation']['raw_output'],'Synthetic software fixture 0.'))
    (out/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in parsed))
    refresh_run(session,key)
    bundle=tmp_path/'failed-review';review.make_bundle(session,bundle,True)
    return bundle


@pytest.mark.parametrize('categories',[['no_issue_identified'],['insufficient_information'],['polarity_contradiction'],[review.TECHNICAL_DISPOSITION,'other_entailment_failure'],[]])
def test_technical_failure_cannot_be_counted_as_semantic(session,failed_bundle,tmp_path,categories):
    rows=review_rows(failed_bundle);cases=rt.read(failed_bundle/'blinded/cases.json')['cases']
    code=next(c['review_id'] for c in cases if c['conditions'][0]['technical_status']!='valid')
    for r in rows:
        if r['review_id']==code:r.update(categories=categories,notes='Synthetic technical failure test')
    path=tmp_path/'bad-technical-review.json';rt.write(path,rows)
    with pytest.raises(ValueError):review.validate_review(session,failed_bundle,path,True)


def test_valid_cell_cannot_abstain_from_semantic_review(session,bundle,tmp_path):
    rows=review_rows(bundle);rows[0].update(categories=[review.TECHNICAL_DISPOSITION],notes='Test')
    path=tmp_path/'invalid-disposition.json';rt.write(path,rows)
    with pytest.raises(ValueError):review.validate_review(session,bundle,path,True)


def test_technical_disposition_keeps_denominators_separate(session,failed_bundle,tmp_path):
    rows=review_rows(failed_bundle);cases=rt.read(failed_bundle/'blinded/cases.json')['cases']
    code=next(c['review_id'] for c in cases if c['conditions'][0]['technical_status']!='valid')
    for row in rows:
        if row['review_id']==code:row.update(categories=[review.TECHNICAL_DISPOSITION],notes='No structured proposal; fabricated parse error')
    path=tmp_path/'complete-review.json';rt.write(path,rows)
    review.finalize(session,failed_bundle,path,tmp_path/'final',True)
    results=rt.read(tmp_path/'final/review_results.json')['condition_results']
    assert sum(r['reviewed'] for r in results)==240
    assert sum(r['semantically_reviewed'] for r in results)==228
    assert sum(r['technical_failure_unreviewable'] for r in results)==12
    assert sum(r.get('no_issue_identified',0) for r in results)==228
    assert sum(r.get('insufficient_information',0) for r in results)==0
    comparisons=rt.read(tmp_path/'final/organizer_comparison.json')['rows']
    invalid=[r for r in comparisons if r['technical_status']!='valid']
    assert len(invalid)==12 and all(r['entailment_categories']==[] and r['review_disposition']==review.TECHNICAL_DISPOSITION for r in invalid)


def test_direct_backend_also_blocked_before_loading():
    with pytest.raises(PermissionError):rt.real_backend(None,None)


def test_qwen_eos_and_secondary_reparse_are_enforced(session):
    key=rt.ORDER[0];out=session/'session'/key
    raw=rt.lines(out/'raw.jsonl');raw[0]['generation']['output_token_ids']=[8,1]
    (out/'raw.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in raw));refresh_run(session,key)
    with pytest.raises(ValueError):rt.verify_session(session,True)


def test_secondary_cannot_be_silently_changed(session):
    key=rt.ORDER[0];out=session/'session'/key
    parsed=rt.lines(out/'predictions.jsonl');parsed[0]['secondary_response'][0]['extracted_label']='negative'
    (out/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in parsed));refresh_run(session,key)
    with pytest.raises(ValueError):rt.verify_session(session,True)


def test_attempt_ledger_is_mandatory(session):
    key=rt.ORDER[0];out=session/'session'/key
    attempts=rt.lines(out/'attempts.jsonl');attempts.pop()
    (out/'attempts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in attempts));refresh_run(session,key)
    with pytest.raises(ValueError):rt.verify_session(session,True)


def test_partial_recovery_counts_dispatch_intent_before_crash(tmp_path,monkeypatch):
    prepared,expected,fd=worker_fixture(tmp_path)
    class Crash(rt.SyntheticBackend):
        def generate(self,encoded):raise RuntimeError('synthetic crash before output')
    monkeypatch.setattr(rt,'SyntheticBackend',Crash)
    assert not rt.run_once(prepared,expected,rt.ORDER[0],worker_fd=fd)
    recovery=rt.inspect_partial(prepared)
    assert recovery['durable_attempt_starts']==1 and recovery['not_recorded_as_started']==19
    assert recovery['runs'][0]['raw_records']==recovery['runs'][0]['parsed_records']==0
    assert not recovery['partial_records_are_verified_predictions']


def test_partial_tail_is_flagged_not_parsed(session):
    p=session/'session'/rt.ORDER[0]/'predictions.jsonl'
    with p.open('a') as stream:stream.write('{"case_index":')
    result=rt.inspect_partial(session)
    assert result['runs'][0]['incomplete_tails']['predictions.jsonl']
    with pytest.raises(ValueError):rt.verify_session(session,True)


def test_confidence_is_part_of_primary_repeat_gate(session):
    # Synthetic valid evidence, unequal confidence; no LLM or semantic claim.
    for key,confidence in [('control-1',0.5),('control-2',0.6)]:
        out=session/'session'/key;raw=rt.lines(out/'raw.jsonl');parsed=rt.lines(out/'predictions.jsonl')
        obj=json.loads(raw[0]['generation']['raw_output'])
        obj['ACL']=dict(label='uncertain',evidence_text='Synthetic software fixture 0.',confidence=confidence)
        text=json.dumps(obj);raw[0]['generation'].update(raw_output=text,decoded_with_special_tokens=text)
        parsed[0]=dict(case_index=0,raw_record_sha256=rt.digest(raw[0]),
            response=rt.validate_primary(text,'Synthetic software fixture 0.'),
            secondary_response=rt.validate_secondary(text,'Synthetic software fixture 0.'))
        for name,rows in [('raw.jsonl',raw),('predictions.jsonl',parsed)]:
            (out/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
        refresh_run(session,key)
    summary=review.technical_summary(session,True)
    gate=next(g for g in summary['technical_gates'] if g['arm']=='control')
    assert gate['valid_identical_cells']==59 and not gate['passed']


def test_review_finalize_never_loads_reference_csv(session,bundle,tmp_path,monkeypatch):
    original=rt.read
    def checked(path):
        assert Path(path).name not in ('details.json','review_context.json','qwen_error_review.csv','train.csv')
        return original(path)
    monkeypatch.setattr(rt,'read',checked)
    path=tmp_path/'review.json';rt.write(path,review_rows(bundle))
    review.finalize(session,bundle,path,tmp_path/'result',True)
    result=rt.read(tmp_path/'result/organizer_comparison.json')
    assert result['status']=='DEFERRED_NO_REFERENCE_LABELS_LOADED'
    assert all(row['organizer'] is None for row in result['rows'])
