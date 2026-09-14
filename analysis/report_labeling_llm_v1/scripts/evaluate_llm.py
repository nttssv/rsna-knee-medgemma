"""Evaluate verified local model outputs; technical failures remain in every full denominator."""
import argparse
import json
from pathlib import Path
import pandas as pd
from benchmark_core import (state_dir, private_path, new_directory, checked_source, verify_run, verify_freeze,
                            LABELS, sha, write_json, utc, config, reviewed_languages)
from analysis_metrics import checked_frame, metrics, agreement_frames, language_metrics, paired_bootstrap


def tables(state, prepared, medgemma_run, qwen_run, partition, variant, freeze_path=None):
    gold = {r['StudyInstanceUID']: r for r in checked_source(state, include_gold=True) if r['split'] == partition}
    languages = reviewed_languages(state)
    frames = {}; manifests = {}
    for name, run in [('medgemma', medgemma_run), ('qwen', qwen_run)]:
        manifest, predictions = verify_run(state, run, name, partition, prepared)
        if partition == 'validation' and manifest['freeze_sha256'] != sha(freeze_path):
            raise ValueError('Validation run uses a different execution freeze')
        manifests[name] = manifest
        rows = []
        for study in predictions:
            reference = gold[study['StudyInstanceUID']]
            for value in study[variant]:
                r = dict(value, StudyInstanceUID=study['StudyInstanceUID'], language=languages[study['StudyInstanceUID']],
                         language_heuristic=reference['language'],
                         organizer_label=reference['gold'][value['condition']])
                rows.append(r)
        frames[name] = checked_frame(pd.DataFrame(rows), gold)
    snapshot = state / 'runs/report-labeling-20260913-v1'
    baseline_manifest = json.loads((snapshot / 'evaluation_manifest.json').read_text())
    baseline_path = snapshot / 'long_extractions.csv'
    if sha(baseline_path) != baseline_manifest['output_sha256']['long_extractions.csv']:
        raise ValueError('Frozen baseline predictions changed')
    base = pd.read_csv(baseline_path, dtype={'StudyInstanceUID': str}, keep_default_na=False)
    base = base[base.split == partition].copy()
    base['status'] = 'valid'
    base['language_heuristic'] = base.StudyInstanceUID.map({uid: row['language'] for uid, row in gold.items()})
    base['language'] = base.StudyInstanceUID.map(languages)
    frames['rule_v1'] = checked_frame(base, gold)
    columns = ['StudyInstanceUID', 'condition', 'organizer_label']
    for frame in frames.values():
        if not frames['rule_v1'][columns].equals(frame[columns]):
            raise ValueError('Gold label or join mismatch between methods')
    return frames, manifests


def disagreement_table(frames, source):
    rows = []
    for i in range(len(frames['rule_v1'])):
        reference = frames['rule_v1'].iloc[i]
        uid, condition, gold = reference.StudyInstanceUID, reference.condition, int(reference.organizer_label)
        row = {'StudyInstanceUID': uid, 'condition': condition, 'organizer_label': gold,
               'language': reference.language, 'report': source[uid]['Report']}
        needs_review = False
        for name, frame in frames.items():
            r = frame.iloc[i]; pred = {'positive': 1, 'negative': 0}.get(r.extracted_label)
            row.update({name + '_prediction': r.extracted_label, name + '_evidence': r.evidence_text,
                        name + '_status': r.status})
            needs_review |= pred != gold
        if needs_review:
            statuses = [frame.iloc[i].status for frame in frames.values()]
            row['error_category'] = 'technical_failure' if any(s != 'valid' for s in statuses) else 'manual_review_required'
            row['review_status'] = 'unreviewed_not_clinical_adjudication'
            rows.append(row)
    columns = ['StudyInstanceUID', 'condition', 'organizer_label', 'language', 'report', 'error_category', 'review_status']
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state-dir'); p.add_argument('--prepared', required=True)
    p.add_argument('--medgemma-run', required=True); p.add_argument('--qwen-run', required=True)
    p.add_argument('--partition', choices=['development', 'validation'], default='validation')
    p.add_argument('--freeze'); p.add_argument('--output', required=True)
    a = p.parse_args(); state = state_dir(a.state_dir)
    if a.partition == 'validation':
        if not a.freeze:
            raise ValueError('Validation evaluation needs the original LLM execution freeze')
        verify_freeze(private_path(state, a.freeze), a.prepared)
    out = new_directory(state, a.output)
    source = {r['StudyInstanceUID']: r for r in checked_source(state) if r['split'] == a.partition}
    for variant in ['first_pass', 'repair_assisted']:
        frames, manifests = tables(state, a.prepared, a.medgemma_run, a.qwen_run, a.partition, variant, a.freeze)
        folder = out / variant; folder.mkdir(mode=0o700)
        parts = []
        for name, frame in frames.items():
            table = metrics(frame); table.insert(0, 'model', name); parts.append(table)
            table.to_csv(folder / (name + '_' + a.partition + '_metrics.csv'), index=False)
        pd.concat(parts, ignore_index=True).to_csv(folder / 'model_comparison.csv', index=False)
        agreed, diagnostic = agreement_frames(frames)
        pd.concat([metrics(frame).assign(strategy=name) for name, frame in agreed.items()], ignore_index=True).to_csv(folder / 'agreement_metrics.csv', index=False)
        diagnostic.to_csv(folder / 'agreement_state_counts.csv', index=False)
        language_metrics(frames).to_csv(folder / 'language_metrics.csv', index=False)
        overall, deltas = paired_bootstrap(dict(rule_v1=frames['rule_v1'], medgemma=frames['medgemma'], qwen=frames['qwen'], **agreed))
        overall.to_csv(folder / 'overall_metrics.csv', index=False); deltas.to_csv(folder / 'paired_deltas.csv', index=False)
        errors = disagreement_table(frames, source)
        errors.to_csv(folder / 'private_disagreements.csv', index=False)
        errors.groupby(['error_category', 'review_status']).size().reset_index(name='condition_checks').to_csv(folder / 'error_category_counts.csv', index=False)
        failure_counts = pd.concat([frame.groupby('status').size().reset_index(name='condition_checks').assign(model=name) for name, frame in frames.items()], ignore_index=True)
        failure_counts.to_csv(folder / 'failure_counts.csv', index=False)
        pd.DataFrame([{'condition': c, 'classification': 'manual_review_required',
                       'scaling_supported': False, 'reason': 'reused_small_validation_requires_fresh_adjudicated_validation'} for c in LABELS]).to_csv(folder / 'condition_recommendations.csv', index=False)
    for file in out.rglob('*.csv'):
        file.chmod(0o600)
    write_json(out / 'evaluation_manifest.json', {'completed_utc': utc(), 'partition': a.partition,
        'interpretation': 'exploratory_reused_validation', 'primary_variant': 'first_pass',
        'recommendation': 'E_more_manual_annotation_and_fresh_validation_required',
        'model_runs': manifests, 'source_sha256': config('benchmark')['source_sha256'],
        'files': {str(p.relative_to(out)): sha(p) for p in out.rglob('*.csv')}})
    print('Saved verified comparison to private state. Public aggregation requires an explicit file/privacy review.')


if __name__ == '__main__':
    main()
