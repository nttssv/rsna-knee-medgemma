"""Check both official tokenizers on all 58 inputs before any model inference."""
import argparse
import json
from pathlib import Path
from benchmark_core import (state_dir, load_inputs, new_directory, config, code_hashes,
                            prompt_for, write_json, write_jsonl, sha, text_sha)


def check(state, prepared, output, allow_download=False):
    from model_runtime import Tokenizer
    rows = load_inputs(state, prepared, 'development') + load_inputs(state, prepared, 'validation')
    output = new_directory(state, output)
    summary = {'code_sha256': code_hashes(), 'models': {}, 'inference_started': False}
    for name in ['medgemma', 'qwen']:
        encoder = Tokenizer(name, state / 'cache/huggingface', allow_download)
        rendered = []
        for row in rows:
            text = encoder.render(prompt_for(name, row['Report']))
            length = encoder.encode(text)['input_ids'].shape[-1]
            rendered.append({'StudyInstanceUID': row['StudyInstanceUID'], 'split': row['split'],
                             'rendered_prompt': text, 'rendered_prompt_sha256': text_sha(text),
                             'input_tokens': length, 'within_input_limit': length <= config(name)['max_input_tokens']})
        write_jsonl(output / (name + '.jsonl'), rendered)
        write_json(output / (name + '_tokenizer.json'), encoder.metadata())
        summary['models'][name] = {'studies': len(rows), 'overflow_studies': sum(not r['within_input_limit'] for r in rendered),
                                  'max_input_tokens_observed': max(r['input_tokens'] for r in rendered)}
    summary['files'] = {p.name: sha(p) for p in output.glob('*') if p.is_file()}
    write_json(output / 'preflight_manifest.json', summary)
    return summary


def verify_preflight(path):
    path = Path(path)
    m = json.loads((path / 'preflight_manifest.json').read_text())
    if m['code_sha256'] != code_hashes() or set(m['models']) != {'medgemma', 'qwen'}:
        raise ValueError('Tokenizer preflight predates current code/config/prompt')
    for model in m['models'].values():
        if model['studies'] != 58:
            raise ValueError('Tokenizer preflight must cover all 58 studies')
    for name, digest in m['files'].items():
        if Path(name).name != name or sha(path / name) != digest:
            raise ValueError('Tokenizer preflight fingerprint changed')
    return m


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state-dir'); p.add_argument('--prepared', required=True); p.add_argument('--output', required=True)
    p.add_argument('--execute', action='store_true'); p.add_argument('--allow-download', action='store_true')
    a = p.parse_args()
    if not a.execute:
        print('DRY RUN: no tokenizer downloads or tokenization. Add --execute only after resource approval.')
    else:
        m = check(state_dir(a.state_dir), a.prepared, a.output, a.allow_download)
        print(json.dumps(m['models'], indent=2))
