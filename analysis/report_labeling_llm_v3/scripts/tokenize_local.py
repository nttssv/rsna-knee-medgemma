"""CPU tokenizer sizing only. Requires cached assets; never loads model weights."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys
import v3_candidate as v3


def run(prepared,cache,output):
    prepared,output=Path(prepared).resolve(),Path(output).resolve()
    if output.exists() or not output.is_relative_to(prepared):
        raise ValueError('Use a new tokenizer-output directory within prepared private state')
    v3.verify_history()
    plan=json.loads((prepared/'plan.json').read_text())
    if plan['candidate_sha256']!=v3.fingerprints():
        raise ValueError('Prepared v3 fingerprints are stale')
    for name,digest in plan['files'].items():
        if v3.core.sha(prepared/name)!=digest:
            raise ValueError('Prepared file changed')
    cfg=v3.experiment()
    snapshot=Path(cache)/('models--'+cfg['model_id'].replace('/','--'))/'snapshots'/cfg['revision']
    assets=json.loads((v3.V2/'aggregate/medgemma_pinned_preflight.json').read_text())['assets']
    for asset in assets:
        if v3.core.sha(snapshot/asset['file'])!=asset['sha256']:
            raise ValueError('Pinned local tokenizer/config asset changed')
    sys.path.insert(0,str(v3.V2/'scripts'))
    from v2_runtime import HFEncoder,check_versions
    versions=check_versions()
    encoder=HFEncoder('medgemma',cache)
    model_config=json.loads((snapshot/'config.json').read_text())
    context=model_config['text_config']['max_position_embeddings']
    entries=[];arms={}
    for arm in cfg['arms']:
        prompts=[json.loads(x) for x in (prepared/(arm+'_prompts.jsonl')).read_text().splitlines()]
        lengths=[]
        for record in prompts:
            encoded=encoder.encode(record['prompt'])
            if encoded.input_tokens>cfg['max_input_tokens'] or encoded.input_tokens+cfg['max_new_tokens']>context:
                raise ValueError('Input/output budget exceeds declared bounds')
            lengths.append(encoded.input_tokens)
            entries.append({'arm':arm,'case_index':record['case_index'],'rendered_prompt':encoded.rendered_prompt,
                            'input_token_ids':encoded.input_ids,'input_tokens':encoded.input_tokens})
        arms[arm]={'reports':len(lengths),'min_input_tokens':min(lengths),'max_input_tokens':max(lengths),
                   'max_input_plus_output_budget':max(lengths)+cfg['max_new_tokens']}
    output.mkdir(mode=0o700)
    details=output/'tokenized_inputs.json';details.write_text(json.dumps(entries,ensure_ascii=False)+'\n');details.chmod(0o600)
    summary={'status':'TOKENIZED_NO_INFERENCE','plan_sha256':v3.core.sha(prepared/'plan.json'),
             'candidate_sha256':v3.fingerprints(),'package_versions':versions,'arms':arms,
             'context_limit':context,'model_calls':0,'weights_loaded':False,'gpu_used':False,
             'template_token_parity':True,'execution_ready':False,'details_sha256':v3.core.sha(details)}
    receipt=output/'receipt.json';receipt.write_text(json.dumps(summary,indent=2)+'\n');receipt.chmod(0o600)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','cache','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();result=run(a.prepared,a.cache,a.output)
    print(json.dumps({'status':result['status'],'arms':result['arms'],'model_calls':0}))
