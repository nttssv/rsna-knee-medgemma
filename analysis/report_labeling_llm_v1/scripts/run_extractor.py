"""Run a pinned text-only extractor into a new private directory; dry-run by default."""
import argparse
import json
from pathlib import Path
import time
from benchmark_core import (state_dir, private_path, new_directory, load_inputs, config,
    code_hashes, verify_freeze, extract_case, prompt_for, sha, utc, write_json, write_jsonl, read_jsonl, text_sha, failure)


def main(model_name):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state-dir'); p.add_argument('--prepared', required=True); p.add_argument('--output', required=True)
    p.add_argument('--partition', choices=['development', 'validation'], default='development')
    p.add_argument('--limit', type=int, help='Development smoke test only; validation always uses all 18')
    p.add_argument('--tokenizer-preflight'); p.add_argument('--freeze')
    p.add_argument('--execute', action='store_true'); p.add_argument('--allow-download', action='store_true')
    a = p.parse_args(); state = state_dir(a.state_dir)
    rows = load_inputs(state, a.prepared, a.partition)
    private_path(state, a.output)
    if a.limit is not None:
        if a.partition != 'development' or not 1 <= a.limit <= 40:
            raise ValueError('Limits are only permitted for 1–40 development studies')
        rows = rows[:a.limit]
    if not a.execute:
        print(json.dumps({'status': 'dry_run_no_models_loaded', 'model': config(model_name),
                          'partition': a.partition, 'studies': len(rows), 'unlabeled_studies': 0}, indent=2))
        return
    if not a.tokenizer_preflight:
        raise ValueError('Both-model tokenizer preflight is required before inference')
    from tokenizer_preflight import verify_preflight
    preflight = private_path(state, a.tokenizer_preflight)
    verify_preflight(preflight)
    if a.partition == 'validation':
        if not a.freeze:
            raise ValueError('Validation requires a completed development freeze')
        verify_freeze(private_path(state, a.freeze), a.prepared)
    out = new_directory(state, a.output)
    manifest = {'model_key': model_name, 'model_config': config(model_name), 'partition': a.partition,
                'started_utc': utc(), 'code_sha256': code_hashes(), 'studies_requested': len(rows),
                'source_sha256': config('benchmark')['source_sha256'], 'split_sha256': config('benchmark')['split_sha256'],
                'preflight_sha256': sha(preflight / 'preflight_manifest.json'),
                'freeze_sha256': sha(a.freeze) if a.freeze else None,
                'interpretation': 'exploratory_reused_validation', 'adapter_loaded': False,
                'images_used': False, 'organizer_labels_in_model_input': False, 'unlabeled_studies_processed': 0}
    write_json(out / 'started_manifest.json', manifest)
    from hardware_audit import audit
    write_json(out / 'hardware.json', audit(state))
    from model_runtime import Tokenizer, HFGenerator
    encoder = Tokenizer(model_name, state / 'cache/huggingface', a.allow_download)
    expected = {r['StudyInstanceUID']: r for r in read_jsonl(preflight / (model_name + '.jsonl'))}
    for row in rows:
        if text_sha(encoder.render(prompt_for(model_name, row['Report']))) != expected[row['StudyInstanceUID']]['rendered_prompt_sha256']:
            raise ValueError('Official rendered prompt changed since tokenizer preflight')
    generator = HFGenerator(encoder, state / 'cache/huggingface', a.allow_download)
    write_json(out / 'tokenizer.json', encoder.metadata())
    write_json(out / 'generation_config.json', generator.gen.to_dict())
    predictions = []
    start = time.perf_counter()
    for row in rows:
        if time.perf_counter() - start > config(model_name)['max_run_seconds']:
            result = {'first_pass': failure('run_time_budget_exceeded'), 'repair_assisted': failure('run_time_budget_exceeded'),
                      'attempts': [], 'repair_attempted': False, 'repair_changed_labels': 0}
        else:
            result = extract_case(generator, prompt_for(model_name, row['Report']), row['Report'])
        record = {k: row[k] for k in ['StudyInstanceUID', 'split', 'report_sha256', 'language']}
        record.update(result)
        # Save each completed case immediately; interrupted runs remain visibly incomplete.
        write_json(out / ('case_' + str(len(predictions)).zfill(3) + '.json'), record)
        predictions.append(record)
        print(f'{model_name}: completed {len(predictions)}/{len(rows)} studies', flush=True)
    write_jsonl(out / 'predictions.jsonl', predictions)
    manifest.update(status='completed', completed_utc=utc(), studies_processed=len(rows),
                    runtime_seconds=time.perf_counter() - start,
                    output_sha256={'predictions.jsonl': sha(out / 'predictions.jsonl')},
                    peak_gpu_allocated_gib=generator.torch.cuda.max_memory_allocated() / 2**30)
    write_json(out / 'run_manifest.json', manifest)
