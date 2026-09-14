"""Paired study-level comparisons with explicit medical abstentions and technical failures."""
import numpy as np
import pandas as pd
from benchmark_core import LABELS, BINARY, STATES, baseline_metrics_module, config


def checked_frame(frame, expected_ids):
    expected = {(uid, c) for uid in expected_ids for c in LABELS}
    keys = list(zip(frame.StudyInstanceUID, frame.condition))
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Every study must have exactly one row for each of 12 conditions')
    if not frame.organizer_label.isin([0, 1]).all():
        raise ValueError('Invalid organizer labels')
    valid = frame.status.eq('valid')
    if not frame.loc[valid, 'extracted_label'].isin(STATES).all():
        raise ValueError('Valid outputs require one of the four medical states')
    if frame.loc[~valid, 'extracted_label'].notna().any():
        raise ValueError('Failed or rejected outputs must not contain a medical label')
    return frame.sort_values(['StudyInstanceUID', 'condition']).reset_index(drop=True)


def metrics(frame):
    baseline = baseline_metrics_module()
    output = []
    for condition, group in frame.groupby('condition', sort=False):
        r = baseline.condition_metrics(group)
        r.update(condition=condition, predicted_positive=int(group.extracted_label.eq('positive').sum()),
                 predicted_negative=int(group.extracted_label.eq('negative').sum()),
                 technical_failures=int((~group.status.isin(['valid', 'selection_abstention'])).sum()),
                 selection_abstentions=int(group.status.eq('selection_abstention').sum()))
        r['conditional_error_rate'] = 1 - r['accuracy'] if r['accuracy'] is not None else None
        r['abstention_rate'] = 1 - r['coverage']
        r['all_study_correct_label_yield_ci_low'], r['all_study_correct_label_yield_ci_high'] = baseline.wilson(r['TP'] + r['TN'], r['studies'])
        if sum(r[k] for k in ['predicted_positive', 'predicted_negative', 'uncertain', 'not_mentioned', 'technical_failures', 'selection_abstentions']) != len(group):
            raise ValueError('Failure/abstention accounting does not cover the full denominator')
        output.append(r)
    return pd.DataFrame(output)


def agreement_frames(frames):
    rule, med, qwen = [frames[k] for k in ['rule_v1', 'medgemma', 'qwen']]
    for other in [med, qwen]:
        if not rule[['StudyInstanceUID', 'condition', 'organizer_label']].equals(other[['StudyInstanceUID', 'condition', 'organizer_label']]):
            raise ValueError('Agreement inputs must be aligned on identical study/condition/reference keys')
    r, m, q = [f.extracted_label for f in [rule, med, qwen]]
    rm = r.isin(BINARY) & r.eq(m); rq = r.isin(BINARY) & r.eq(q)
    mq = m.isin(BINARY) & m.eq(q)
    masks = {'all_three_binary': rm & rq, 'medgemma_qwen_binary': mq,
             'rule_medgemma_binary': rm, 'rule_qwen_binary': rq, 'rule_at_least_one_llm_binary': rm | rq}
    result = {}
    for name, mask in masks.items():
        f = rule.copy()
        labels = m if name == 'medgemma_qwen_binary' else r
        f['extracted_label'] = labels.where(mask, None)
        f['status'] = np.where(mask, 'valid', 'selection_abstention')
        f['confidence'] = 0.0  # No confidence score is defined for model agreement.
        result[name] = f
    diagnostic = []
    for a, b in [('rule_v1', 'medgemma'), ('rule_v1', 'qwen'), ('medgemma', 'qwen')]:
        x, y = frames[a], frames[b]
        for state in sorted(STATES):
            diagnostic.append({'model_a': a, 'model_b': b, 'matching_state': state,
                               'count': int((x.extracted_label.eq(state) & y.extracted_label.eq(state)).sum()),
                               'usable_binary_agreement': state in BINARY})
    return result, pd.DataFrame(diagnostic)


def language_metrics(frames):
    rows = []
    languages = sorted(set().union(*(set(f.language) for f in frames.values())) | {'Bulgarian', 'Croatian', 'Dutch', 'German'})
    for name, frame in frames.items():
        for language in languages:
            group = frame[frame.language == language]
            if group.empty:
                rows.append({'model': name, 'language': language, 'condition': 'ALL', 'studies': 0,
                             'interpretation': 'not_represented'})
            else:
                for r in metrics(group).to_dict('records'):
                    r.update(model=name, language=language, language_method='recorded_post_baseline_analyst_review_not_new_detection',
                             interpretation='descriptive_only_small_n')
                    rows.append(r)
    return pd.DataFrame(rows)


def cluster_samples(frame):
    # Each row below is one study; all twelve correlated condition checks stay together.
    p = frame.extracted_label.map({'positive': 1, 'negative': 0}).to_numpy(dtype=float).reshape(-1, 12)
    y = frame.organizer_label.to_numpy(dtype=float).reshape(-1, 12)
    return p, y


def summaries(p, y):
    decided = np.isfinite(p)
    correct = decided & (p == y)
    axis = tuple(range(1, p.ndim))
    n = decided.sum(axis=axis)
    c = correct.sum(axis=axis)
    total = np.prod(p.shape[1:])
    accuracy = np.divide(c, n, out=np.full(n.shape, np.nan), where=n > 0)
    return {'coverage': n / total, 'all_study_correct_label_yield': c / total,
            'conditional_error_rate': 1 - accuracy}


def paired_bootstrap(frames, replicates=None, seed=None):
    if not frames:
        raise ValueError('No models for bootstrap')
    names = list(frames)
    base = frames[names[0]]
    for frame in frames.values():
        if not base[['StudyInstanceUID', 'condition', 'organizer_label']].equals(frame[['StudyInstanceUID', 'condition', 'organizer_label']]):
            raise ValueError('Bootstrap must pair identical studies, conditions and labels')
    n = base.StudyInstanceUID.nunique()
    cfg = config('benchmark')
    repetitions = replicates or cfg['bootstrap_replicates']
    rng = np.random.default_rng(seed if seed is not None else cfg['bootstrap_seed'])
    draws = rng.integers(0, n, size=(repetitions, n))
    point, sample = {}, {}
    for name, frame in frames.items():
        p, y = cluster_samples(frame)
        point[name] = summaries(p[None, ...], y[None, ...])
        sample[name] = summaries(p[draws], y[draws])
    rows = []
    for name in names:
        for metric in point[name]:
            values = sample[name][metric]
            finite = values[np.isfinite(values)]
            low, high = np.quantile(finite, [.025, .975]) if len(finite) else (np.nan, np.nan)
            rows.append(dict(model=name, metric=metric, estimate=point[name][metric][0], ci_low=low, ci_high=high,
                             valid_bootstrap_draws=len(finite), requested_draws=repetitions, studies=n,
                             interpretation='exploratory_study_cluster_percentile_interval'))
    delta_rows = []
    for name in names:
        if name == 'rule_v1':
            continue
        for metric in point[name]:
            values = sample[name][metric] - sample['rule_v1'][metric]
            finite = values[np.isfinite(values)]
            low, high = np.quantile(finite, [.025, .975]) if len(finite) else (np.nan, np.nan)
            delta_rows.append(dict(model=name, reference='rule_v1', metric=metric,
                delta=point[name][metric][0] - point['rule_v1'][metric][0], ci_low=low, ci_high=high,
                valid_bootstrap_draws=len(finite), requested_draws=repetitions, studies=n,
                interpretation='exploratory_paired_delta_no_confirmatory_test'))
    return pd.DataFrame(rows), pd.DataFrame(delta_rows)
