"""Offline forensics of saved MedGemma responses. Never changes v2 predictions."""
import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'diagnostics'))
sys.path.insert(0, str(ROOT / 'scripts'))
from build_partial_review import read_verified
from core import config, sha, validate_response
from run_smoke import private_path
from v2_runtime import HFEncoder, messages_for

OPEN, CLOSE = '<unused94>', '<unused95>'


def strict_suffix(raw):
    """One exact leading, nonempty envelope; do not strip or repair the suffix."""
    if not raw.startswith(OPEN) or raw.count(OPEN) != 1 or raw.count(CLOSE) != 1:
        raise ValueError('Expected exactly one leading balanced envelope')
    end = raw.index(CLOSE)
    if not raw[len(OPEN):end].strip():
        raise ValueError('Empty thought envelope')
    suffix = raw[end + len(CLOSE):]
    if not suffix.strip():
        raise ValueError('Missing answer suffix')
    return suffix


def hypothetical_response(raw, report, status):
    # Preserve generation status: a truncated response is never rescued.
    return validate_response(strict_suffix(raw), report, status)


def token_structure(ids, opener, closer, eos):
    starts = [i for i, value in enumerate(ids) if value == opener]
    ends = [i for i, value in enumerate(ids) if value == closer]
    eos_positions = [i for i, value in enumerate(ids) if value in eos]
    terminal = bool(ids and ids[-1] in eos)
    balanced = starts == [0] and len(ends) == 1 and ends[0] > 1
    if not balanced or eos_positions != ([len(ids)-1] if terminal else []):
        raise ValueError('Unexpected delimiter/EOS structure')
    close = ends[0]
    return dict(output_tokens=len(ids), opening_positions=starts, closing_positions=ends,
                thought_tokens=close-1, delimiter_tokens=2,
                answer_tokens=len(ids)-close-1-int(terminal),
                eos_tokens=int(terminal), eos_positions=eos_positions,
                last_token_id=ids[-1], terminal_eos=terminal)


def verify_assets(cache):
    cfg = config('medgemma')
    snapshot = Path(cache) / ('models--' + cfg['model_id'].replace('/', '--')) / 'snapshots' / cfg['revision']
    baseline = json.loads((ROOT / 'aggregate/medgemma_pinned_preflight.json').read_text())
    assets = baseline['assets']
    for entry in assets:
        path = snapshot / entry['file']
        if path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
            raise ValueError('Pinned tokenizer/config asset differs: ' + entry['file'])
    return snapshot, {entry['file']: entry['sha256'] for entry in assets}


def run(prepared, cache, output):
    output = private_path(output)
    if output.exists():
        raise FileExistsError('Refuse to overwrite prior diagnostics')
    cases, original = read_verified(prepared)
    snapshot, assets = verify_assets(cache)
    encoder = HFEncoder('medgemma', cache)  # Processor/tokenizer only; no HFBackend.
    tokenizer = encoder.tokenizer
    from jinja2 import Environment, meta
    template_variables = sorted(meta.find_undeclared_variables(Environment().parse(encoder.processor.chat_template)))
    markers = {}
    for token in (OPEN, CLOSE):
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError('Delimiter is not one token')
        markers[token] = dict(id=ids[0], is_special=ids[0] in tokenizer.all_special_ids)
    eos = config('runtime')['models']['medgemma']['generation_eos_token_ids']
    prompts = [json.loads(x) for x in (Path(prepared) / 'medgemma_prompts.jsonl').read_text().splitlines()]
    controls = []
    for i, case in enumerate(cases):
        prompt = prompts[i]['prompt']
        base = encoder.encode(prompt)
        matches = {}
        # Negative controls only: these kwargs are not claimed as supported.
        for value in (False, True):
            rendered = encoder.processor.apply_chat_template(messages_for('medgemma', prompt),
                tokenize=False, add_generation_prompt=True, enable_thinking=value)
            ids = tokenizer.encode(rendered, add_special_tokens=False)
            matches[str(value)] = rendered == base.rendered_prompt and ids == base.input_ids
        if case['runs']:
            saved = case['runs']['medgemma-1']
            if base.rendered_prompt != saved['rendered_prompt'] or base.input_ids != saved['input_token_ids']:
                raise ValueError('Current pinned prompt rendering differs from executed input')
        controls.append(dict(case_index=case['case_index'], unsupported_kwarg_parity=matches))
    details, counts = [], Counter()
    for case in cases[:3]:
        record = case['runs']['medgemma-1']
        ids = record['output_token_ids']
        decode = lambda values: tokenizer.decode(values, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        body = ids[:-1] if ids[-1] in eos else ids
        if decode(ids) != record['decoded_with_special_tokens'] or decode(body) != record['raw_output']:
            raise ValueError('Saved token/text parity failed')
        structure = token_structure(ids, markers[OPEN]['id'], markers[CLOSE]['id'], eos)
        suffix = strict_suffix(record['raw_output'])
        if suffix != decode(body[structure['closing_positions'][0]+1:]):
            raise ValueError('Token suffix differs from unchanged text suffix')
        hypothetical = hypothetical_response(record['raw_output'], case['report'], record['generation_status'])
        row_counts = Counter(row['status'] for row in hypothetical['rows'])
        counts.update(row_counts)
        details.append(dict(case_index=case['case_index'], language=case['language'],
            report=case['report'], raw_output=record['raw_output'], answer_suffix=suffix,
            organizer=case['gold'], structure=structure,
            generation_status=record['generation_status'], official_response=record['response'],
            hypothetical_response=hypothetical, hypothetical_status_counts=dict(row_counts)))
    summary = dict(status='OFFLINE_HYPOTHETICAL_DIAGNOSTIC_ONLY', model_id=config('medgemma')['model_id'],
        revision=config('medgemma')['revision'], source_run_commit=original['execution_source_commit'],
        reviewed_plan_sha256=original['plan_sha256'], model_calls=0, weights_loaded=False, gpu_used=False,
        official_v2_accepted_cells=0, official_v2_status=original['status'],
        markers=markers, template_variables=template_variables,
        unsupported_enable_thinking_true_false_unchanged_on_all_five=all(all(c['unsupported_kwarg_parity'].values()) for c in controls),
        saved_prompt_parity_on_three_attempts=True, saved_raw_token_parity_on_three_attempts=True,
        assets_sha256=assets,
        package_versions={name: version(name) for name in ('torch','transformers','tokenizers','jinja2','huggingface-hub')},
        generation_config=json.loads((snapshot/'generation_config.json').read_text()),
        hypothetical_status_counts=dict(counts),
        cases=[{k:d[k] for k in ('case_index','language','structure','generation_status','hypothetical_status_counts')} for d in details],
        evidence_validity_is_not_semantic_correctness=True,
        code_sha256={p.name:sha(p) for p in HERE.iterdir() if p.suffix in ('.py','.html')})
    output.mkdir(mode=0o700, parents=True)
    for name, value in [('summary.json', summary), ('private_details.json',details), ('template_controls.json',controls)]:
        path = output / name
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
        path.chmod(0o600)
    payload = json.dumps({'summary':summary, 'cases':details}, ensure_ascii=False)
    for char in '<>&':
        payload = payload.replace(char, '\\u%04x' % ord(char))
    (output/'response_diagnostic.html').write_text((HERE/'viewer.html').read_text().replace('__PAYLOAD__',payload))
    (output/'response_diagnostic.html').chmod(0o600)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','cache','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.prepared, args.cache, args.output)
    print(json.dumps({'status':summary['status'], 'hypothetical_status_counts':summary['hypothetical_status_counts'], 'model_calls':0}))
