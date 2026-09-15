"""Synthetic integration checks only; generated artifacts live in pytest temp dirs."""
import importlib.util
import json
from pathlib import Path
import sys
import pytest

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
import core
sys.path.insert(0,str(Path(__file__).parent))
from session_fixture import attach_session
spec=importlib.util.spec_from_file_location('v2_smoke_review',SCRIPTS/'smoke_review.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)


def setup_prepared(tmp_path):
    prepared=tmp_path/'prepared';prepared.mkdir()
    cases=[dict(case_index=i,StudyInstanceUID=f'synthetic-{i}',report='🙂 ACL\nis intact.',language='Synthetic',gold={c:0 for c in core.LABELS},rule=[dict(condition=c,extracted_label='not_mentioned') for c in core.LABELS]) for i in range(5)]
    inputs=[dict(StudyInstanceUID=c['StudyInstanceUID'],Report=c['report']) for c in cases]
    (prepared/'inputs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in inputs))
    (prepared/'review_context.json').write_text(json.dumps(cases))
    (prepared/'manifest.json').write_text(json.dumps(dict(candidate_code_sha256=core.code_hashes(),files={p.name:core.sha(p) for p in prepared.iterdir()})))
    return prepared,cases


def test_not_run_preview_has_no_invented_scores_and_escapes_html(tmp_path):
    prepared,_=setup_prepared(tmp_path)
    result=v.build(prepared,prepared/'preview')
    assert result['status']=='NOT RUN' and result['measured_model_runs']==0
    assert (prepared/'preview/smoke_metrics.csv').read_text()=='status\nNOT RUN\n'
    html=(prepared/'preview/case_viewer.html').read_text()
    assert 'Array.from(c.report)' in html and 'NOT RUN' in html
    assert (prepared/'preview/smoke_dashboard.png').stat().st_size>0
    with pytest.raises(ValueError):v.build(prepared,prepared/'preview')


def test_future_run_contract_checks_hashes_cases_and_saved_parser(tmp_path):
    prepared,cases=setup_prepared(tmp_path);runs=prepared/'adapter_runs';runs.mkdir()
    for key in v.KEYS:
        folder=runs/key;folder.mkdir();model=key.rsplit('-',1)[0]
        obj={c:dict(label='not_mentioned',evidence_text='',confidence=0) for c in core.LABELS}
        obj['ACL']=dict(label='negative',evidence_text='ACL is intact.',confidence=.8)
        raw=json.dumps(obj)
        records=[dict(StudyInstanceUID=c['StudyInstanceUID'],raw_output=raw,generation_status='completed',response=core.validate_response(raw,c['report']),rendered_prompt='SYNTHETIC FIXTURE ONLY',runtime_seconds=1,input_tokens=20,output_tokens=10) for c in cases]
        (folder/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        m=dict(status='completed',model_key=model,model_revision=core.config(model)['revision'],prepared_manifest_sha256=core.sha(prepared/'manifest.json'),candidate_code_sha256=core.code_hashes(),predictions_sha256=core.sha(folder/'predictions.jsonl'),peak_gpu_allocated_gib=None)
        (folder/'run_manifest.json').write_text(json.dumps(m))
    attach_session(prepared,runs)
    _,metrics,_=v.load(prepared,runs)
    assert all(r['valid']==60 and r['decided']==5 and r['correct']==5 for r in metrics)
    path=runs/'medgemma-1/run_manifest.json';manifest=json.loads(path.read_text())
    manifest['synthetic']=True;path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):v.load(prepared,runs)
    manifest['synthetic']=False;path.write_text(json.dumps(manifest))
    attach_session(prepared,runs)
    parent_path=runs/'session_manifest.json';parent=json.loads(parent_path.read_text())
    parent['results'].reverse();parent_path.write_text(json.dumps(parent))
    with pytest.raises(ValueError,match='reordered'):v.load(prepared,runs)
    parent['results'].reverse();parent_path.write_text(json.dumps(parent))
    path=runs/'qwen-1/run_manifest.json';m=json.loads(path.read_text());m['session_id']='wrong-session';path.write_text(json.dumps(m))
    with pytest.raises(ValueError):v.load(prepared,runs)
    attach_session(prepared,runs)
    (runs/'qwen-2/predictions.jsonl').write_text('tampered')
    with pytest.raises(ValueError,match='hash mismatch'):v.load(prepared,runs)


def test_stale_candidate_rejected(tmp_path):
    prepared,_=setup_prepared(tmp_path)
    path=prepared/'manifest.json';m=json.loads(path.read_text());m['candidate_code_sha256']={};path.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='stale'):v.load(prepared)


@pytest.mark.parametrize('count',[0,4])
def test_incomplete_prepared_inputs_rejected(tmp_path,count):
    prepared,_=setup_prepared(tmp_path)
    p=prepared/'inputs.jsonl';p.write_text('\n'.join(p.read_text().splitlines()[:count]))
    m=json.loads((prepared/'manifest.json').read_text());m['files']['inputs.jsonl']=core.sha(p)
    (prepared/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='fixed five'):v.load(prepared)
