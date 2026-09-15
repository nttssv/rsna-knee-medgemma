"""Read-only diagnostic of the pinned failed RTX smoke; never a benchmark gate."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from core import sha, validate_response
from smoke_review import load, KEYS


def read_verified(prepared):
    """Bind all original artifacts to the inventory copied before provider stop."""
    prepared = Path(prepared).resolve()
    summary = json.loads((ROOT / 'aggregate/rtx_smoke.json').read_text())
    inventory = prepared / 'execution_environment/remote-file-hashes.json'
    if sha(inventory) != summary['remote_inventory_sha256']:
        raise ValueError('Not the recorded RTX artifact inventory')
    hashes = json.loads(inventory.read_text())
    if len(hashes) != 17:
        raise ValueError('Incomplete inventory')
    for name, digest in hashes.items():
        path = (prepared / name).resolve()
        if not path.is_relative_to(prepared) or sha(path) != digest:
            raise ValueError('Original artifact changed: ' + name)
    cases, _, _ = load(prepared)  # Existing preparation/code/split verification.
    session = json.loads((prepared / 'adapter_runs/session_manifest.json').read_text())
    if session['status'] != 'failed':
        raise ValueError('This diagnostic only supports the recorded failed session')
    directory = prepared / 'adapter_runs/medgemma-1'
    raw = [json.loads(x) for x in (directory / 'raw_generations.jsonl').read_text().splitlines()]
    predictions = [json.loads(x) for x in (directory / 'predictions.jsonl').read_text().splitlines()]
    if len(raw) != 3 or len(predictions) != 3:
        raise ValueError('Unexpected partial run length')
    for case, source, prediction in zip(cases, raw, predictions):
        if source['StudyInstanceUID'] != case['StudyInstanceUID'] or source['synthetic']:
            raise ValueError('Input identity or backend mismatch')
        response = validate_response(source['raw_output'], case['report'], source['generation_status'])
        if prediction != dict(source, response=response):
            raise ValueError('Saved prediction differs from exact raw-response parsing')
        case['runs']['medgemma-1'] = prediction
    return cases, summary


def build(prepared, output):
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    if output.exists() or not output.is_relative_to(prepared):
        raise ValueError('Choose a new output folder inside private prepared state')
    cases, summary = read_verified(prepared)
    payload = json.dumps({'cases': cases, 'summary': summary, 'runs': KEYS}, ensure_ascii=False)
    for char in '<>&':
        payload = payload.replace(char, '\\u%04x' % ord(char))
    template = (Path(__file__).parent / 'partial_viewer.html').read_text()
    output.mkdir(mode=0o700)
    target = output / 'case_viewer.html'
    target.write_text(template.replace('__PAYLOAD__', payload))
    target.chmod(0o600)
    return {'status': summary['status'], 'attempted_reports': 3, 'accepted_labels': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.prepared, args.output)))
