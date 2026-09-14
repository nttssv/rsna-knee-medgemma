"""Compare repeated inference on the first five development studies; never read gold."""
import argparse
from pathlib import Path
from benchmark_core import state_dir, verify_run, code_hashes, private_path, write_json, sha


def signature(row):
    return {variant: [{k: x[k] for k in ['condition', 'extracted_label', 'evidence_text', 'status']}
                      for x in row[variant]] for variant in ['first_pass', 'repair_assisted']}


def compare(first, second):
    if len(first) < 5 or len(second) < 5:
        raise ValueError('Need at least five development studies per repeat')
    if [r['StudyInstanceUID'] for r in first[:5]] != [r['StudyInstanceUID'] for r in second[:5]]:
        raise ValueError('Repeatability studies do not match')
    stable = all(signature(a) == signature(b) for a, b in zip(first[:5], second[:5]))
    # Repeated technical failure is not evidence that inference works.
    technically_valid = all(x['status'] == 'valid' for rows in [first[:5], second[:5]] for r in rows for x in r['first_pass'])
    return {'stable_labels_evidence_status': stable, 'all_first_pass_cells_valid': technically_valid,
            'passed': stable and technically_valid}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state-dir'); p.add_argument('--prepared', required=True); p.add_argument('--output', required=True)
    for model in ['medgemma', 'qwen']:
        p.add_argument('--' + model + '-first', required=True); p.add_argument('--' + model + '-repeat', required=True)
    a = p.parse_args(); state = state_dir(a.state_dir); results = {}; hashes = {}
    for model in ['medgemma', 'qwen']:
        paths = [Path(getattr(a, model + '_' + suffix)) for suffix in ['first', 'repeat']]
        if paths[0].resolve() == paths[1].resolve():
            raise ValueError('Repeatability requires two distinct inference runs')
        runs = [verify_run(state, path, model, 'development', a.prepared, allow_subset=True)[1] for path in paths]
        results[model] = compare(*runs)
        hashes[model] = [sha(path / 'run_manifest.json') for path in paths]
    result = {'code_sha256': code_hashes(), 'models_checked': ['medgemma', 'qwen'],
              'studies_per_model': 5, 'results': results, 'run_manifest_sha256': hashes,
              'passed': all(r['passed'] for r in results.values())}
    write_json(private_path(state, a.output), result)
    print('Repeatability passed:', result['passed'])
