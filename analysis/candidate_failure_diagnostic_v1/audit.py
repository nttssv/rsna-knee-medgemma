"""Read-only audit of the recorded v3 failure. No model loading or generation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
EXEC = REPO / 'analysis/report_labeling_llm_v3_execution_v1'
sys.path.insert(0, str(EXEC / 'diagnostics'))
from build_dashboard import load_verified, rt
from v2_runtime import HFEncoder, check_versions, generation_status

INVENTORY_SHA = '19f31b50f361f911cc3f139f8a96dddadd0cfd377a3450644f1368bccbf83ea7'
PLAN_SHA = '6bcbee08e0bdb30f148f6a7a84c75afdcd327dc45c94b03b82724f0b625f1ee1'


def require(value, message):
    if not value:
        raise ValueError(message)


def verify_history():
    manifest = rt.read(ROOT / 'protected_history.json')
    for name, digest in manifest['files'].items():
        require(rt.sha(REPO / name) == digest, 'Historical source changed: ' + name)
    return len(manifest['files'])


def audit_assets(cache):
    baseline = rt.read(REPO / 'analysis/report_labeling_llm_v2/aggregate/medgemma_pinned_preflight.json')
    cfg = rt.model_config()
    snapshot = Path(cache) / ('models--' + cfg['model_id'].replace('/', '--')) / 'snapshots' / cfg['revision']
    for item in baseline['assets']:
        path = snapshot / item['file']
        require(path.stat().st_size == item['bytes'] and rt.sha(path) == item['sha256'], 'Pinned tokenizer asset mismatch')
    return {x['file']: x['sha256'] for x in baseline['assets']}


def audit_receipts(prepared, plan, parent):
    folder = prepared / 'session'
    start = rt.read(folder / 'start.json')
    require(parent['status'] == 'failed' and parent['synthetic'] is False and parent['error_type'] is None, 'Wrong parent result')
    require(parent['session_id'] == start['session_id'] and parent['start_sha256'] == rt.sha(folder/'start.json') and parent['plan_sha256'] == PLAN_SHA == start['plan_sha256'], 'Parent binding mismatch')
    require(start['run_order'] == rt.ORDER and start['synthetic'] is False, 'Wrong run order/backend')
    for name in ('authorization', 'watchdog'):
        require(start[name+'_sha256'] == rt.sha(folder/(name+'.json')), 'Grant/watchdog snapshot mismatch')
    grant, watch = rt.read(folder/'authorization.json'), rt.read(folder/'watchdog.json')
    require(grant['plan_sha256'] == PLAN_SHA and grant['proposal_sha256'] == rt.sha(EXEC/'configs/resource_proposal.json') and grant['approved'] is True, 'Grant binding mismatch')
    require(watch['pod_id'] == grant['pod_id'] and watch['deadline'] == grant['provider_deadline'] == start['provider_deadline'] and watch['status'] == 'armed' and watch['cli_syntax_and_read_access_verified'] is True and watch['script_sha256'] == rt.sha(EXEC/'scripts/stop_watchdog.py'), 'Watchdog binding mismatch')
    require([x['key'] for x in parent['results']] == ['control-1','candidate-1'], 'Unexpected dispatch outcome sequence')
    require({p.name for p in folder.glob('dispatch-*.json')} == {'dispatch-0.json','dispatch-1.json'}, 'Unexpected later dispatch')
    require(not any((folder/k).exists() for k in ('candidate-2','control-2')), 'Unexpected later worker')
    manifest = rt.read(REPO/'analysis/report_labeling_llm_v2/configs/weight_manifest.json')['medgemma']
    receipts = []
    for entry, count in zip(parent['results'], (5,4)):
        key=entry['key']; out=folder/key; result=rt.read(out/'result.json'); ctx=rt.context(prepared,key)
        require(entry['receipt_sha256'] == rt.sha(out/'result.json'), 'Child receipt hash mismatch')
        wanted='completed' if key=='control-1' else 'failed'
        require(result['status']==entry['status']==wanted and result['attempted']==count and result['synthetic'] is False and all(result.get(k)==v for k,v in ctx.items()), 'Child receipt/context mismatch')
        require(entry['process_status']==('completed' if count==5 else 'child_failed'), 'Child process status mismatch')
        required={'cache_audit.json','preflight.json','runtime.json','raw.jsonl','predictions.jsonl'}
        require(set(result['artifacts'])==required and all(rt.sha(out/n)==h for n,h in result['artifacts'].items()), 'Child artifact binding mismatch')
        cache=rt.read(out/'cache_audit.json')
        require(cache['complete'] is True and cache['model']=='medgemma' and cache['revision']==manifest['revision'], 'Cache receipt identity mismatch')
        expected=[dict(file=f['file'],bytes=f['bytes'],expected_bytes=f['bytes'],digest=f['digest'],verified=True) for f in manifest['files']]
        require(cache['files']==expected, 'Cache receipt does not cover all pinned files')
        runtime=rt.read(out/'runtime.json'); backend=runtime['backend']; pre=rt.read(out/'preflight.json')
        require(runtime['model']==plan['model']==backend['effective_v3_model_config'], 'Model configuration mismatch')
        require(runtime['encoder']==pre['encoder'] and all(backend.get(k)==v for k,v in pre['encoder'].items()), 'Encoder metadata mismatch')
        require(all(backend['package_versions'][k].split('+')[0]==v for k,v in rt.core.config('runtime')['required_versions'].items()), 'Package pin mismatch')
        require(backend['gpu_name']=='NVIDIA RTX 6000 Ada Generation' and backend['nvidia_driver_version']=='570.124.06' and backend['cuda_version']=='12.8' and backend['dtype']=='bfloat16' and backend['attention']=='sdpa' and backend['seed']==20260914, 'Execution metadata mismatch')
        gen=backend['effective_generation_config']
        require(all(gen[k]==v for k,v in dict(max_new_tokens=4096,max_time=300,do_sample=False,num_beams=1,eos_token_id=[1,106],pad_token_id=0).items()), 'Decoding configuration mismatch')
        raw=rt.lines(out/'raw.jsonl'); parsed=rt.lines(out/'predictions.jsonl')
        require(len(raw)==len(parsed)==count and len(pre['input_tokens'])==5, 'Recorded count mismatch')
        statuses=[r['generation']['generation_status'] for r in raw]
        require(statuses==(['completed']*5 if count==5 else ['completed']*3+['generation_truncated']), 'Wrong stopping boundary')
        require(pre['input_tokens'][:count]==[r['generation']['input_tokens'] for r in raw], 'Preflight input count mismatch')
        require(rt.parse_time(start['started_at'])<=rt.parse_time(result['started_at'])<rt.parse_time(result['finished_at'])<=rt.parse_time(parent['finished_at'])<rt.parse_time(grant['provider_deadline']), 'Receipt timeline mismatch')
        require(0<result['elapsed_seconds']<=1800 and result['elapsed_seconds']>=sum(r['generation']['runtime_seconds'] for r in raw), 'Child elapsed time mismatch')
        receipts.append(dict(run=key,recorded=count,receipt_and_cache_metadata_verified=True))
    rt.verify_run(prepared,'control-1')
    return receipts


def token_roundtrip(encoder, source, record):
    encoded=encoder.encode(record['prompt']);g=record['generation'];ids=g['output_token_ids']
    require(encoded.rendered_prompt==record['rendered_prompt'] and encoded.input_ids==record['input_ids'], 'Input render/token roundtrip mismatch')
    require(len(encoded.input_ids)==g['input_tokens'] and len(ids)==g['output_tokens'] and all(type(x) is int and x>=0 for x in ids), 'Token count/type mismatch')
    body=ids[:-1] if ids and ids[-1] in (1,106) else ids
    require(encoder.decode(ids)==g['decoded_with_special_tokens'] and encoder.decode(body)==g['raw_output'], 'Output decode/EOS roundtrip mismatch')
    require(generation_status(ids,[1,106],g['runtime_seconds'],300,4096)==g['generation_status'], 'Generation status mismatch')
    return True


def token_layout(ids, opening, closing):
    starts=[i for i,n in enumerate(ids) if n==opening];ends=[i for i,n in enumerate(ids) if n==closing]
    terminal=int(bool(ids and ids[-1] in (1,106)));total=len(ids)-terminal
    if starts==[0] and len(ends)==1 and ends[0]>0:
        thought=ends[0]-1;answer=total-ends[0]-1;kind='balanced_thought_then_answer'
    elif starts==[0] and not ends:
        thought=total-1;answer=0;kind='unfinished_thought'
    elif not starts and not ends:
        thought=0;answer=total;kind='no_recorded_thought_envelope'
    else:thought=answer=None;kind='other_delimiter_layout'
    grams=Counter(tuple(ids[i:i+16]) for i in range(max(0,len(ids)-15)))
    return dict(output_tokens=len(ids),opening_positions=starts,closing_positions=ends,
        thought_tokens=thought,answer_surface_tokens=answer,terminal_eos_tokens=terminal,
        layout=kind,max_identical_16_token_window_occurrences=max(grams.values(),default=0),
        repeated_16_token_window_excess=sum(n-1 for n in grams.values()),
        repetition_note='Overlapping exact windows; descriptive repetition evidence, not a causal explanation')


def surface_structure(raw):
    """Count exact visible lines only; do not rescue labels from a partial array."""
    lines=[line.strip() for line in raw.splitlines() if line.strip().startswith('{"evidence_text"')]
    counts=Counter(lines)
    return dict(starts_with_fenced_array=raw.startswith('```json\n['),
        object_shaped_lines=len(lines),distinct_object_shaped_lines=len(counts),
        most_frequent_exact_object_line_count=max(counts.values(),default=0),
        labels_rescued=0)


def json_shape(response):
    try:obj=json.loads(response['normalized_output'],object_pairs_hook=rt.core.no_duplicates)
    except (ValueError,TypeError):return dict(parseable=False)
    if not isinstance(obj,dict):return dict(parseable=True,top_level_type=type(obj).__name__)
    values=list(obj.values());states={'positive','negative','uncertain','not_mentioned'}
    return dict(parseable=True,exact_condition_keys=set(obj)==set(rt.core.LABELS),
        value_type_counts=dict(Counter(type(v).__name__ for v in values)),
        string_state_values=sum(isinstance(v,str) and v in states for v in values),
        other_string_values=sum(isinstance(v,str) and v not in states for v in values),
        converted_to_labels=False)


def run(prepared,cache,inventory,output):
    output=rt.private(output);require(not output.exists(),'Choose a new private output directory')
    history=verify_history();prepared=rt.private(prepared)
    require(rt.sha(prepared/'plan.json')==PLAN_SHA,'Wrong recorded plan')
    plan,parent,outputs=load_verified(prepared,inventory,INVENTORY_SHA)
    receipts=audit_receipts(prepared,plan,parent)
    assets=audit_assets(cache);versions=check_versions();encoder=HFEncoder('medgemma',cache)
    marker_ids=[encoder.tokenizer.encode(t,add_special_tokens=False) for t in ('<unused94>','<unused95>')]
    require(all(len(x)==1 for x in marker_ids),'Unexpected marker tokenization')
    inputs=rt.lines(prepared/'inputs.jsonl');details=[]
    for key in ('control-1','candidate-1'):
        raw=rt.lines(prepared/'session'/key/'raw.jsonl');parsed=rt.lines(prepared/'session'/key/'predictions.jsonl')
        pre=rt.read(prepared/'session'/key/'preflight.json')
        all_enc=[encoder.encode(rt.candidate.make_prompt(key.split('-')[0],s['Report'])) for s in inputs]
        require([e.input_tokens for e in all_enc]==pre['input_tokens'] and encoder.metadata==pre['encoder'],'All-five preflight reconstruction mismatch')
        for i,(record,pred) in enumerate(zip(raw,parsed)):
            token_roundtrip(encoder,inputs[i],record)
            details.append(dict(run=key,case_index=i,input_and_output_roundtrip=True,
                generation_status=record['generation']['generation_status'],
                token_layout=token_layout(record['generation']['output_token_ids'],marker_ids[0][0],marker_ids[1][0]),
                shape=json_shape(pred['response']),surface=surface_structure(record['generation']['raw_output']),raw_output_sha256=rt.digest(record['generation']['raw_output'])))
    summary=dict(status='OFFLINE_AUDIT_PASSED',audit_source_sha256=rt.sha(Path(__file__)),protocol_sha256=rt.sha(ROOT/'PROTOCOL.md'),model_loaded=False,generations=0,organizer_labels_evaluated=False,
        protected_files=history,recorded_calls_audited=len(details),input_roundtrips=9,output_roundtrips=9,
        all_ten_preflight_input_counts_reconstructed=True,child_receipts=receipts,
        tokenizer_assets=assets,local_package_versions=versions,
        short_candidate_shapes=[x['shape'] for x in details if x['run']=='candidate-1' and x['generation_status']=='completed'],
        truncated_candidate_layout=[x['token_layout'] for x in details if x['generation_status']!='completed'][0],
        truncated_candidate_surface=[x['surface'] for x in details if x['generation_status']!='completed'][0],
        control_thought_tokens_total=sum(x['token_layout']['thought_tokens'] for x in details if x['run']=='control-1'),
        control_answer_tokens_total=sum(x['token_layout']['answer_surface_tokens'] for x in details if x['run']=='control-1'),
        historical_predictions_changed=False,failed_comparison_still_failed=True,expansion_allowed=False,
        limitations=['Receipt audits validate saved evidence, not an independent replay of GPU computation or historical live process identity.','No model weights rehashed locally; the original complete 13-file cache audit receipt and its inventory binding were verified.','Token provenance does not establish medical correctness or prove why the model chose its output.'])
    output.mkdir(mode=0o700);rt.write(output/'details.json',details);rt.write(output/'summary.json',summary)
    require(verify_history()==history,'History changed during audit')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('prepared','cache','inventory','output'):p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args();summary=run(a.prepared,a.cache,a.inventory,a.output)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('tokenizer_assets','local_package_versions')},indent=2))
