"""Local v3 preparation and response contract. No inference implementation."""
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
V2 = ROOT.parent / 'report_labeling_llm_v2'
spec = importlib.util.spec_from_file_location('v3_frozen_v2_core', V2/'scripts/core.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
OPEN, CLOSE = '<unused94>', '<unused95>'
REVIEW_CATEGORIES = ['polarity_contradiction','wrong_anatomy_or_compartment','severity_threshold_error',
    'temporality_error','unsupported_negative','other_entailment_failure','reference_discordance',
    'insufficient_information','no_issue_identified']


def experiment():
    return json.loads((ROOT/'configs/experiment.json').read_text())


def fingerprints():
    paths = [ROOT/'PROTOCOL.md', ROOT/'ontology_equivalence.csv']
    for directory in ('scripts','configs','prompts','tests'):
        paths += [p for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    return {str(p.relative_to(ROOT)):core.sha(p) for p in sorted(paths)}


def verify_history():
    protected = json.loads((ROOT/'configs/protected_history.json').read_text())['files']
    for name, digest in protected.items():
        if core.sha(REPO/name) != digest:
            raise ValueError('Protected historical file changed: '+name)
    for name, original in [('control_v2.txt','medgemma_prompt_v2.txt'),
                           ('target_definitions.txt','target_definitions_v2.txt')]:
        if (ROOT/'prompts'/name).read_bytes() != (V2/'prompts'/original).read_bytes():
            raise ValueError('Control content or target definitions changed')
    return len(protected)


def make_prompt(arm, report):
    instruction = (ROOT/'prompts'/experiment()['arms'][arm]['prompt']).read_text()
    definitions = (ROOT/'prompts/target_definitions.txt').read_text()
    return instruction+'\nTARGET DEFINITIONS:\n'+definitions+'\nREPORT_JSON_STRING:\n'+json.dumps(report,ensure_ascii=False)


def validate_v3(raw, report, generation_status='completed'):
    """Shared prospective framing; frozen v2 lexical checks, no label correction."""
    surface, removed = raw, False
    if generation_status == 'completed' and (OPEN in raw or CLOSE in raw):
        valid = raw.startswith(OPEN) and raw.count(OPEN)==1 and raw.count(CLOSE)==1
        if valid:
            end = raw.index(CLOSE)
            valid = bool(raw[len(OPEN):end].strip()) and bool(raw[end+len(CLOSE):].strip())
        if not valid:
            response = core.validate_response(raw,report,'envelope_error')
            response.update(generation_status=generation_status,parser_status='envelope_error',
                            thought_envelope_removed=False,answer_surface=raw)
            return response
        surface, removed = raw[end+len(CLOSE):], True
    response = core.validate_response(surface,report,generation_status)
    response.update(raw_output=raw,thought_envelope_removed=removed,answer_surface=surface)
    return response


def readiness():
    return {'status':'NOT_EXECUTION_READY','execution_allowed':False,
            'reasons':['No v3 GPU adapter exists','No measured v3 outputs or qualified semantic review',
                       'No current resource quote or execution authorization for this candidate']}


def prepare(state, source, output):
    state, source, output = map(lambda p:Path(p).resolve(), (state,source,output))
    if state.is_relative_to(REPO) and not state.is_relative_to(REPO/'state'):
        raise ValueError('Private state in this checkout must be under state/')
    if not source.is_relative_to(state) or not output.is_relative_to(state) or output==state or output.exists():
        raise ValueError('Choose an existing private source and a new output under private state')
    cfg = experiment()
    if cfg['execution_enabled'] or cfg['gpu_adapter_implemented']:
        raise ValueError('This package supports local preparation only')
    protected_count = verify_history()
    if core.sha(source/'manifest.json') != cfg['source_prepared_manifest_sha256']:
        raise ValueError('Expected the anchored v2 preparation manifest')
    manifest = json.loads((source/'manifest.json').read_text())
    if manifest['candidate_code_sha256'] != core.code_hashes():
        raise ValueError('Source is not the unchanged v2 prepared candidate')
    required = ['inputs.jsonl','review_context.json','v2_adjudication_candidates.csv']
    for name in required:
        if core.sha(source/name) != manifest['files'][name]:
            raise ValueError('Source preparation changed: '+name)
    if core.sha(source/'inputs.jsonl') != cfg['source_inputs_sha256']:
        raise ValueError('Expected exact five previously reviewed inputs')
    split = state/'runs/report-labeling-20260913-v1/splits.csv'
    if core.sha(split) != manifest['split_sha256']:
        raise ValueError('Original split changed')
    with split.open() as stream:
        assignments = list(csv.DictReader(stream))
    if len(assignments)!=58 or len({r['StudyInstanceUID'] for r in assignments})!=58 or [sum(r['split']==s for r in assignments) for s in ('development','validation')]!=[40,18]:
        raise ValueError('Expected original unique 40/18 membership')
    inputs = [json.loads(line) for line in (source/'inputs.jsonl').read_text().splitlines()]
    dev = {r['StudyInstanceUID'] for r in assignments if r['split']=='development'}
    if len(inputs)!=5 or not {r['StudyInstanceUID'] for r in inputs} <= dev:
        raise ValueError('Only five fixed development reports permitted')
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent,prefix='.v3-prepare-') as temp:
        tmp=Path(temp)
        for name in required:
            shutil.copyfile(source/name,tmp/name)
        for arm in cfg['arms']:
            records=[{'case_index':i,'prompt':make_prompt(arm,r['Report'])} for i,r in enumerate(inputs)]
            (tmp/(arm+'_prompts.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
        fields=['run_key','case_index','condition','output_sha256','reviewer','reviewed_at','categories_json','notes']
        with (tmp/'semantic_review_template.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
            for run in cfg['run_order']:
                for i in range(5):
                    for condition in core.LABELS:
                        writer.writerow(dict(run_key=run,case_index=i,condition=condition))
        plan={'status':'DRY_DESIGN_ONLY','model_calls':0,'run_order':cfg['run_order'],
              'studies':5,'maximum_primary_generations':20,'experiment':cfg,'readiness':readiness(),
              'candidate_sha256':fingerprints(),'source_manifest_sha256':core.sha(source/'manifest.json'),
              'split_sha256':core.sha(split),'files':{p.name:core.sha(p) for p in tmp.iterdir()}}
        (tmp/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
        for p in tmp.iterdir():p.chmod(0o600)
        tmp.rename(output)
    output.chmod(0o700)
    return {'status':'NOT_RUN','model_calls':0,'studies':5,'protected_files':protected_count,
            'plan_sha256':core.sha(output/'plan.json'),'readiness':readiness()}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-dir',type=Path,default=Path(os.environ.get('RSNA_STATE_DIR',REPO/'state')))
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.state_dir,args.source,args.output),indent=2))
