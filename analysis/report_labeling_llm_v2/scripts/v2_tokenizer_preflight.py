"""Local tokenization only: no backend, generation, weights, or provider API."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from core import code_hashes, config, prompt_for, sha
from run_smoke import checked_inputs, private_path, write_json
from v2_runtime import HFEncoder, check_versions, encode_prompt, text_sha

ASSETS = ('tokenizer.json', 'tokenizer.model', 'tokenizer_config.json',
          'special_tokens_map.json', 'added_tokens.json', 'processor_config.json',
          'preprocessor_config.json', 'chat_template.jinja', 'config.json',
          'generation_config.json')


def inventory(directory, require_hub_blobs=False):
    """Hash only small allowlisted assets; never read weight files."""
    directory = Path(directory)
    records = []
    for name in ASSETS:
        p = directory/name
        if not p.exists():
            continue
        if p.stat().st_size > 64 * 1024**2:
            raise ValueError('Tokenizer/config asset exceeds 64 MiB limit')
        data = p.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        blob = p.resolve().name if p.is_symlink() else None
        if require_hub_blobs:
            git_blob = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            if blob not in (digest, git_blob):
                raise ValueError('Cache file does not match its content-addressed Hub blob')
        records.append(dict(file=name, bytes=len(data), sha256=digest, hub_blob=blob))
    if not any(r['file'] == 'tokenizer.json' for r in records):
        raise ValueError('Full tokenizer vocabulary is missing')
    return records


def token_metadata(tokenizer):
    return dict(tokenizer_class=type(tokenizer).__name__, vocab_size=tokenizer.vocab_size,
                total_token_count=len(tokenizer),
                special_tokens_map={k:str(v) for k,v in tokenizer.special_tokens_map.items()},
                all_special_tokens=list(tokenizer.all_special_tokens),
                all_special_ids=list(tokenizer.all_special_ids),
                bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id, unk_token_id=tokenizer.unk_token_id)


def summarize(entries, model, context_limit):
    lengths = [e.input_tokens for e in entries]
    cfg = config(model)
    fits = all(0 < n <= cfg['max_input_tokens'] for n in lengths)
    if not fits:
        raise ValueError('Input token budget exceeded')
    context_fits = (all(n+cfg['max_new_tokens'] <= context_limit for n in lengths)
                    if context_limit is not None else None)
    if context_fits is False:
        raise ValueError('Input plus output budget exceeds config context')
    return dict(studies=len(lengths), min_input_tokens=min(lengths),
                max_input_tokens=max(lengths), total_input_tokens=sum(lengths),
                input_cap=cfg['max_input_tokens'], output_reserve=cfg['max_new_tokens'],
                max_input_plus_output=max(lengths)+cfg['max_new_tokens'],
                input_budget_passed=fits, config_context_limit=context_limit,
                config_context_budget_passed=context_fits,
                rendered_token_parity_passed=True, model_calls=0,
                loaded_weight_context_verified=False)


def run(prepared, model, output, cache=None, local_export=None):
    rows = checked_inputs(prepared)  # Exact five original development reports only.
    output = private_path(output)
    if output.exists():
        raise FileExistsError('Refuse to overwrite a prior preflight')
    if (cache is None) == (local_export is None):
        raise ValueError('Choose exactly one cache or local export')
    if local_export and model != 'medgemma':
        raise ValueError('Local-export diagnostic is limited to the saved MedGemma processor')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
    versions = check_versions()
    cfg = config(model)
    snapshot = (Path(cache)/('models--'+cfg['model_id'].replace('/','--'))/'snapshots'/cfg['revision']
                if cache else Path(local_export))
    assets = inventory(snapshot, require_hub_blobs=bool(cache))
    if cache:
        encoder = HFEncoder(model, cache)
        processor, tokenizer = encoder.processor, encoder.tokenizer
        provenance = 'pinned_hub_snapshot_content_addressed_assets'
    else:
        # Export lacks a recorded revision: do not fabricate a pinned Hub snapshot.
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(str(snapshot.resolve()),
                        local_files_only=True, trust_remote_code=False)
        tokenizer = processor.tokenizer
        provenance = 'historical_local_export_revision_unverified'
    expected = config('runtime')['models'][model]
    if (text_sha(processor.chat_template) != expected['chat_template_sha256'] or
        tokenizer.eos_token_id != expected['tokenizer_eos_token_id'] or
        tokenizer.pad_token_id != expected['tokenizer_pad_token_id']):
        raise ValueError('Template or EOS/PAD differs from pinned expectations')
    entries = [encode_prompt(processor,tokenizer,model,prompt_for(model,r['Report'])) for r in rows]
    context_limit = None
    if cache and (snapshot/'config.json').exists():
        model_config = json.loads((snapshot/'config.json').read_text())
        context_limit = model_config.get('text_config',model_config).get('max_position_embeddings')
        if type(context_limit) is not int or context_limit <= 0:
            raise ValueError('Invalid config context limit')
    aggregate = summarize(entries,model,context_limit)
    output.mkdir(parents=True, mode=0o700)
    write_json(output/'inputs_tokenized.json', [dict(case_index=i,
        rendered_prompt=e.rendered_prompt, input_token_ids=e.input_ids,
        input_tokens=e.input_tokens, rendered_prompt_sha256=text_sha(e.rendered_prompt))
        for i,e in enumerate(entries)])
    result = dict(status='TOKENIZED_NO_INFERENCE', model=model, model_id=cfg['model_id'],
        expected_revision=cfg['revision'], verified_snapshot_revision=cfg['revision'] if cache else None,
        provenance=provenance, candidate_code_sha256=code_hashes(),
        prepared_manifest_sha256=sha(Path(prepared)/'manifest.json'),
        inputs_sha256=sha(Path(prepared)/'inputs.jsonl'), assets=assets,
        chat_template_sha256=text_sha(processor.chat_template),
        package_versions=versions, tokenizer=token_metadata(tokenizer), aggregate=aggregate,
        tokenized_artifact_sha256=sha(output/'inputs_tokenized.json'),
        completed_at=datetime.now(timezone.utc).isoformat(),
        weights_loaded=False, gpu_used=False, inference_ready=False)
    # Detect changes to read-only assets or protected inputs during the check.
    if inventory(snapshot, require_hub_blobs=bool(cache)) != assets:
        raise ValueError('Tokenizer assets changed during preflight')
    checked_inputs(prepared)
    write_json(output/'receipt.json', result)
    print(json.dumps(dict(model=model, provenance=provenance, **aggregate), indent=2))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--model',choices=['medgemma','qwen'],required=True)
    p.add_argument('--output',type=Path,required=True)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--cache',type=Path)
    g.add_argument('--local-export',type=Path)
    a=p.parse_args()
    run(a.prepared,a.model,a.output,a.cache,a.local_export)


if __name__=='__main__':
    main()
