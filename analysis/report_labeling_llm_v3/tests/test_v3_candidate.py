"""Software/fixture expectations only: no measured LLM answers."""
import csv
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import v3_candidate as v3


def output(condition=None,label=None,evidence=None):
    rows={c:{'label':'not_mentioned','evidence_text':'','confidence':0} for c in v3.core.LABELS}
    if condition:
        rows[condition]={'label':label,'evidence_text':evidence,'confidence':0 if label=='not_mentioned' else 1}
    return json.dumps(rows)


def test_control_and_ontology_are_exactly_preserved():
    assert v3.verify_history()==124
    assert v3.make_prompt('control','A synthetic report.')==v3.core.prompt_for('medgemma','A synthetic report.')
    with (v3.ROOT/'ontology_equivalence.csv').open() as f:rows=list(csv.DictReader(f))
    assert [r['condition'] for r in rows]==list(v3.core.LABELS)
    assert all(r['v2_definition']==r['v3_definition'] and r['qualified_clinical_review']=='not_performed' for r in rows)


@pytest.mark.parametrize('wrapper',['{}','```json\n{}\n```','<unused94>thought\ncheck<unused95>{}',
                                     '<unused94>thought\ncheck<unused95>```json\n{}\n```'])
def test_only_shared_framing_changes_acceptance(wrapper):
    raw=wrapper.format(output('ACL','negative','No ACL disruption.'))
    response=v3.validate_v3(raw,'No ACL disruption.')
    assert response['raw_output']==raw
    assert response['rows']==v3.core.validate_response(output('ACL','negative','No ACL disruption.'),'No ACL disruption.')['rows']


@pytest.mark.parametrize('prefix,suffix',[
    ('prose<unused94>thought<unused95>',''),(' <unused94>thought<unused95>',''),
    ('<unused94>unclosed',''),('<unused95>',''),('<unused94><unused95>',''),
    ('<unused94>nested<unused94>x<unused95>',''),('<unused94>x<unused95>','<unused95>'),
])
def test_malformed_envelopes_never_become_labels(prefix,suffix):
    response=v3.validate_v3(prefix+output()+suffix,'')
    assert response['parser_status']=='envelope_error'
    assert all(r['label'] is None for r in response['rows'])


def test_truncation_precedes_envelope_removal():
    raw='<unused94>thought<unused95>'+output()
    response=v3.validate_v3(raw,'','generation_truncated')
    assert not response['thought_envelope_removed']
    assert all(r['status']=='generation_truncated' for r in response['rows'])


def test_no_keyword_medical_fixer_or_semantic_claim():
    response=v3.validate_v3(output('ACL','positive','ACL intact.'),'ACL intact.')
    acl=response['rows'][0]
    assert acl['label']=='positive' and acl['status']=='valid' and acl['semantic_review_required']


FIXTURES=json.loads((v3.ROOT/'tests/semantic_cases.json').read_text())
@pytest.mark.parametrize('case',FIXTURES,ids=lambda c:c['id'])
def test_authored_expectation_is_serializable_not_a_model_result(case):
    assert case['model_response']=='NOT_RUN' and case['qualified_review']=='not_performed'
    response=v3.validate_v3(output(case['condition'],case['expected_label'],case['expected_evidence']),case['report'])
    row=next(r for r in response['rows'] if r['condition']==case['condition'])
    assert row['status']=='valid' and row['label']==case['expected_label']


def source_fixture(tmp_path,monkeypatch):
    state=tmp_path/'state';source=state/'source';source.mkdir(parents=True)
    split=state/'runs/report-labeling-20260913-v1/splits.csv';split.parent.mkdir(parents=True)
    split.write_text('StudyInstanceUID,split\n'+''.join(f's{i},{"development" if i<40 else "validation"}\n' for i in range(58)))
    records=[{'StudyInstanceUID':f's{i}','Report':f'Synthetic report {i}.','split':'development'} for i in range(5)]
    (source/'inputs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    (source/'review_context.json').write_text('[]')
    (source/'v2_adjudication_candidates.csv').write_text('status\nunresolved\n')
    manifest={'candidate_code_sha256':v3.core.code_hashes(),'split_sha256':v3.core.sha(split),
              'files':{p.name:v3.core.sha(p) for p in source.iterdir()}}
    (source/'manifest.json').write_text(json.dumps(manifest))
    cfg=v3.experiment();cfg.update(source_inputs_sha256=v3.core.sha(source/'inputs.jsonl'),source_prepared_manifest_sha256=v3.core.sha(source/'manifest.json'))
    monkeypatch.setattr(v3,'experiment',lambda:cfg)
    return state,source,cfg


def test_preparation_contains_no_model_answers_or_execution_permission(tmp_path,monkeypatch):
    state,source,cfg=source_fixture(tmp_path,monkeypatch)
    target=state/'prepared';result=v3.prepare(state,source,target)
    assert result['model_calls']==0 and not result['readiness']['execution_allowed']
    plan=json.loads((target/'plan.json').read_text())
    assert plan['run_order']==['control-1','candidate-1','candidate-2','control-2']
    with (target/'semantic_review_template.csv').open() as f:reviews=list(csv.DictReader(f))
    assert len(reviews)==240 and all(r['reviewer']==r['output_sha256']=='' for r in reviews)
    prompts=[json.loads(x) for x in (target/'candidate_prompts.jsonl').read_text().splitlines()]
    assert all(set(p)=={'case_index','prompt'} for p in prompts)
    with pytest.raises(ValueError):v3.prepare(state,source,target)


@pytest.mark.parametrize('what',['manifest','inputs','split','execution','outside'])
def test_preparation_refuses_changed_sources_and_unsafe_destination(tmp_path,monkeypatch,what):
    state,source,cfg=source_fixture(tmp_path,monkeypatch)
    target=state/'prepared'
    if what=='manifest':(source/'manifest.json').write_text('{}')
    elif what=='inputs':(source/'inputs.jsonl').write_text('')
    elif what=='split':(state/'runs/report-labeling-20260913-v1/splits.csv').write_text('')
    elif what=='execution':cfg['execution_enabled']=True
    elif what=='outside':target=tmp_path/'public'
    with pytest.raises(ValueError):v3.prepare(state,source,target)
    assert not target.exists()
