"""CPU-only prospective v2 contract. No model imports or semantic diagnosis."""
from pathlib import Path
import hashlib
import json
import math
import re

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
LABELS = ('ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
          'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture')
STATES = {'positive', 'negative', 'uncertain', 'not_mentioned'}
WS = ' \t\r\n'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config(name):
    return json.loads((ROOT / 'configs' / f'{name}.json').read_text())


def verify_protected():
    protected = config('protected_v1')
    for name, digest in protected['files'].items():
        if sha(REPO / name) != digest:
            raise ValueError(f'Protected v1 file changed: {name}')
    return len(protected['files'])


def no_duplicates(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError('duplicate_key')
        obj[key] = value
    return obj


def normalize_output(raw):
    """Preserve captured fence contents exactly, including all internal whitespace."""
    text = raw.strip(WS)
    match = re.fullmatch(r'```(?:json)?\r?\n([\s\S]*?)\r?\n```', text)
    return (match.group(1), 'outer_fence') if match else (raw, 'none')


def collapse_with_offsets(text):
    normalized, spans = [], []
    i = 0
    while i < len(text):
        start = i
        if text[i] in WS:
            while i < len(text) and text[i] in WS:
                i += 1
            normalized.append(' ')
        else:
            normalized.append(text[i])
            i += 1
        spans.append((start, i))
    return ''.join(normalized), spans


def match_evidence(report, evidence):
    if evidence.strip(WS):
        exact, position = [], 0
        while True:
            position = report.find(evidence, position)
            if position < 0: break
            exact.append(position)
            if len(exact)>1:
                return {'status':'ambiguous_evidence_error','reason':'ambiguous_exact_span'}
            position += 1
        if exact:
            start=exact[0]
            return {'status':'valid','source_span':evidence,'evidence_start':start,
                    'evidence_end':start+len(evidence),'evidence_normalization_applied':False,
                    'evidence_normalization_method':'exact','reason':None}
    needle, _ = collapse_with_offsets(evidence.strip(WS))
    if not needle:
        return {'status': 'evidence_error', 'reason': 'empty_evidence'}
    haystack, offsets = collapse_with_offsets(report)
    hits, position = [], 0
    while True:
        position = haystack.find(needle, position)
        if position < 0:
            break
        hits.append(position)
        if len(hits) > 1:
            return {'status': 'ambiguous_evidence_error', 'reason': 'ambiguous_span'}
        position += 1
    if not hits:
        return {'status': 'evidence_error', 'reason': 'no_literal_or_whitespace_match'}
    start = offsets[hits[0]][0]
    end = offsets[hits[0] + len(needle) - 1][1]
    source = report[start:end]
    return {'status': 'valid', 'source_span': source, 'evidence_start': start,
            'evidence_end': end, 'evidence_normalization_applied': source != evidence,
            'evidence_normalization_method': 'ascii_whitespace' if source != evidence else 'exact',
            'reason': None}


def failed(condition, status, reason=None, model_evidence=None):
    return {'condition': condition, 'label': None, 'confidence': None,
            'model_evidence': model_evidence, 'source_span': None,
            'evidence_start': None, 'evidence_end': None,
            'evidence_normalization_applied': False, 'evidence_normalization_method': 'none', 'status': status,
            'reason': reason, 'semantic_review_required': True}


def validate_response(raw, report, generation_status='completed'):
    """Validate syntax and lexical grounding only. No inference of clinical meaning."""
    result = {'raw_output': raw, 'normalized_output': raw, 'normalization_applied': False, 'normalization_type': 'none',
              'generation_status': generation_status, 'parser_status': 'not_parsed', 'rows': []}
    if generation_status != 'completed':
        result['rows'] = [failed(c, generation_status) for c in LABELS]
        return result
    normalized, normalization = normalize_output(raw)
    result.update(normalized_output=normalized, normalization_applied=normalization != 'none', normalization_type=normalization)
    try:
        obj = json.loads(normalized, object_pairs_hook=no_duplicates,
                         parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (ValueError, TypeError) as error:
        status = 'duplicate_key' if str(error) == 'duplicate_key' else 'parse_error'
        result.update(parser_status=status, rows=[failed(c, status) for c in LABELS])
        return result
    if not isinstance(obj, dict) or set(obj) != set(LABELS):
        result.update(parser_status='schema_error', rows=[failed(c, 'schema_error') for c in LABELS])
        return result
    result['parser_status'] = 'valid_json'
    for condition in LABELS:
        value = obj[condition]
        if not isinstance(value, dict) or set(value) != {'label', 'evidence_text', 'confidence'}:
            result['rows'].append(failed(condition, 'schema_error'))
            continue
        label, evidence, confidence = value['label'], value['evidence_text'], value['confidence']
        if (not isinstance(label, str) or label not in STATES or not isinstance(evidence, str)
                or type(confidence) not in (int, float) or not math.isfinite(confidence)
                or not 0 <= confidence <= 1):
            result['rows'].append(failed(condition, 'schema_error'))
            continue
        if label == 'not_mentioned':
            match = ({'status': 'valid', 'source_span': '', 'evidence_start': None,
                      'evidence_end': None, 'evidence_normalization_applied': False, 'evidence_normalization_method':'none', 'reason': None}
                     if evidence == '' and confidence == 0 else
                     {'status': 'evidence_error', 'reason': 'not_mentioned_requires_empty_evidence_zero_confidence'})
        else:
            match = match_evidence(report, evidence)
        if match['status'] != 'valid':
            result['rows'].append(failed(condition, match['status'], match['reason'], evidence))
            continue
        result['rows'].append(dict(condition=condition, label=label, confidence=confidence,
                                   model_evidence=evidence, semantic_review_required=True, **match))
    return result


def prompt_for(model, report):
    cfg = config(model)
    instruction = (ROOT / 'prompts' / cfg['prompt_file']).read_text()
    definitions = (ROOT / 'prompts/target_definitions_v2.txt').read_text()
    return instruction + '\nTARGET DEFINITIONS:\n' + definitions + '\nREPORT_JSON_STRING:\n' + json.dumps(report, ensure_ascii=False)


def code_hashes(root=ROOT):
    paths=[root/'PROTOCOL.md']
    for folder in ('scripts','configs','prompts','tests','smoke_review_tools'):
        paths += [p for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    return {str(p.relative_to(root)):sha(p) for p in sorted(paths)}
