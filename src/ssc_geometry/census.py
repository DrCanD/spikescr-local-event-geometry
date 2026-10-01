"""The complete neighborhood census: outcome partition, source geometry, internal activation
change and clean-activation replacement (manuscript §4.1–4.2 and §4.4–4.6; Tables 3, 5–8;
Figures 2–4).

Everything is regenerated from the released arrays; the reference files are only compared
against, never read into the computation.
"""
from __future__ import annotations
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any
import numpy as np
from . import paths
from .core import classify_transitions, enumerate_moves, validate_input
from .io import load_npz, read_numeric_csv, read_json, write_json, write_csv, compare_tree, compare_internal_statistics
from .statistics import GEOMETRY_OPTIONS, geometry_aggregates, recompute_internal, spearman_rho

GROUPS = ['class_preserved', 'adverse', 'corrective', 'lateral']
BOUNDARY_LABELS = {'stem': 'Stem', 'attention_1': 'Attention 1', 'local_1': 'Local 1', 'block_1': 'Block 1',
                   'attention_2': 'Attention 2', 'local_2': 'Local 2', 'block_2': 'Block 2'}
OUTCOME_LABELS = {'class_preserved': 'Class-preserved', 'class_changed': 'Any class change', 'adverse': 'Adverse',
                  'corrective': 'Corrective', 'lateral': 'Lateral'}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


# ---------------------------------------------------------------- source-level geometry (one row per source)
def derive_transition_masks(label: int, clean_prediction: int, predictions: np.ndarray) -> dict[str, np.ndarray]:
    predictions = np.asarray(predictions, dtype=np.int64)
    changed = predictions != int(clean_prediction)
    clean_correct = int(clean_prediction) == int(label)
    candidate_correct = predictions == int(label)
    adverse = changed & clean_correct & ~candidate_correct
    corrective = changed & (not clean_correct) & candidate_correct
    lateral = changed & ~(adverse | corrective)
    if not np.array_equal(changed, adverse | corrective | lateral):
        raise RuntimeError('Transition partition failed')
    code = np.zeros(len(predictions), dtype=np.uint8)
    code[adverse] = GEOMETRY_OPTIONS['transition_codes']['adverse']
    code[corrective] = GEOMETRY_OPTIONS['transition_codes']['corrective']
    code[lateral] = GEOMETRY_OPTIONS['transition_codes']['lateral']
    return {'changed': changed, 'adverse': adverse, 'corrective': corrective, 'lateral': lateral, 'preserved': ~changed, 'code': code}


def derive_source_geometry(stored: dict[str, Any], arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Recompute one source row of data/neighborhood/sources.csv from its candidate records."""
    label = int(arrays['label']); clean_prediction = int(arrays['clean_prediction'])
    predictions = arrays['predictions'].astype(np.int64)
    from_t = arrays['from_t'].astype(np.int64); to_t = arrays['to_t'].astype(np.int64)
    feature = arrays['feature'].astype(np.int64); multiplicity = arrays['multiplicity'].astype(np.int64)
    masks = derive_transition_masks(label, clean_prediction, predictions)
    direction = to_t - from_t
    horizon = int(stored['horizon_steps'])
    time_norm = from_t / max(1, horizon - 1)
    time_tertile = np.minimum((time_norm * 3).astype(np.int64), 2)
    feature_quartile = np.minimum((feature * 4 // 140).astype(np.int64), 3)
    row: dict[str, Any] = {
        'audit_rank': int(stored['audit_rank']), 'source_index': int(stored['source_index']), 'label': label,
        'clean_prediction': clean_prediction, 'clean_correct': bool(clean_prediction == label),
        'clean_margin': float(stored['clean_margin']), 'horizon_steps': horizon, 'input_count': int(stored['input_count']),
        'unique_candidates': int(len(predictions)), 'event_weighted_candidates': int(multiplicity.sum()),
        'class_change_rate': float(masks['changed'].mean()), 'adverse_rate': float(masks['adverse'].mean()),
        'corrective_rate': float(masks['corrective'].mean()), 'lateral_rate': float(masks['lateral'].mean()),
        'class_changed_unique': int(masks['changed'].sum()), 'adverse_unique': int(masks['adverse'].sum()),
        'corrective_unique': int(masks['corrective'].sum()), 'lateral_unique': int(masks['lateral'].sum()),
        'minimum_clean_class_margin': float(arrays['candidate_clean_class_margin'].min()),
        'minimum_true_class_margin': float(arrays['candidate_true_class_margin'].min()),
        'distinct_candidate_classes': int(len(np.unique(predictions))),
    }
    for value, name in ((-1, 'previous'), (1, 'next')):
        select = direction == value
        row[f'{name}_candidates'] = int(select.sum())
        row[f'{name}_class_change_rate'] = float(masks['changed'][select].mean()) if select.any() else float('nan')
        row[f'{name}_adverse_rate'] = float(masks['adverse'][select].mean()) if select.any() else float('nan')
    row['previous_minus_next_class_change_rate'] = row['previous_class_change_rate'] - row['next_class_change_rate']
    for tertile in range(3):
        select = time_tertile == tertile
        row[f'time_tertile_{tertile + 1}_class_change_rate'] = float(masks['changed'][select].mean()) if select.any() else float('nan')
    for quartile in range(4):
        select = feature_quartile == quartile
        row[f'feature_quartile_{quartile + 1}_class_change_rate'] = float(masks['changed'][select].mean()) if select.any() else float('nan')
    lookup = {(int(t), int(f), int(d)): i for i, (t, f, d) in enumerate(zip(from_t, feature, direction))}
    pair_counts = np.zeros(4, dtype=np.int64)
    for (time_bin, channel, move_direction), previous_index in lookup.items():
        if move_direction != -1:
            continue
        next_index = lookup.get((time_bin, channel, 1))
        if next_index is None:
            continue
        pair_counts[2 * int(masks['changed'][previous_index]) + int(masks['changed'][next_index])] += 1
    row.update({'paired_neither_changed': int(pair_counts[0]), 'paired_next_only_changed': int(pair_counts[1]),
                'paired_previous_only_changed': int(pair_counts[2]), 'paired_both_changed': int(pair_counts[3]),
                'paired_direction_discordance': float((pair_counts[2] - pair_counts[1]) / max(1, pair_counts.sum()))})
    return row


# ---------------------------------------------------------------- benchmark metrics (Table 2 rows)
def benchmark_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    require(labels.ndim == 1 and labels.shape == predictions.shape, 'Benchmark shape mismatch')
    require(all(a.dtype.kind in 'iu' and np.all((a >= 0) & (a < 35)) for a in (labels, predictions)), 'Invalid class IDs')
    confusion = np.zeros((35, 35), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    support, predicted = confusion.sum(axis=1), confusion.sum(axis=0)
    diagonal = np.diag(confusion)
    recall = np.divide(diagonal, support, out=np.zeros(35), where=support > 0)
    precision = np.divide(diagonal, predicted, out=np.zeros(35), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(35), where=(precision + recall) > 0)
    present = support > 0
    return dict(samples=len(labels), correct=int((labels == predictions).sum()), accuracy=float((labels == predictions).mean()),
                balanced_accuracy=float(recall[present].mean()), macro_f1=float(f1[present].mean()),
                worst_class_recall=float(recall[present].min()), class_support=support.tolist(), class_recall=recall.tolist(),
                class_precision=precision.tolist(), class_f1=f1.tolist(), confusion=confusion.tolist())


# ---------------------------------------------------------------- loading with complete cross-checks
def load_and_validate(root: Path) -> dict[str, Any]:
    """Load every census file and check the correspondence of every record, panel source and replacement output."""
    config = read_json(root / paths.EXPERIMENT_CONFIG)
    geometry = load_npz(root / paths.CANDIDATES)
    internal = load_npz(root / paths.INTERNAL)
    inputs = load_npz(root / paths.PANEL_INPUTS)
    clean_file = load_npz(root / paths.PANEL_CLEAN_SCORES)
    panel = read_numeric_csv(root / paths.PANEL_SOURCES)
    stored_source_rows = read_numeric_csv(root / paths.SOURCES)
    n = len(geometry['source_index'])
    require(n == 725070, 'Unexpected candidate count')
    require(len(panel) == 100 and len(stored_source_rows) == 100, 'Expected 100 panel records')
    require(len({row['source_index'] for row in panel}) == 100, 'Duplicate source')
    quotas = Counter(row['label'] for row in panel)
    require(len(quotas) == 35 and set(quotas.values()).issubset({2, 3}), 'Fixed label quota mismatch')
    require(set(inputs) == {f"source_{row['source_index']:05d}" for row in panel}, 'Panel key mismatch')
    require(clean_file['scores'].shape == (100, 35), 'Clean score dimensions')
    require(np.isfinite(clean_file['scores']).all(), 'Nonfinite clean scores')
    require(clean_file['source_index'].tolist() == [row['source_index'] for row in panel], 'Clean source order mismatch')
    clean = {int(s): v for s, v in zip(clean_file['source_index'], clean_file['scores'])}
    for key, arr in geometry.items():
        require(arr.shape == (n,), f'Invalid geometry field shape {key}')
        require(arr.dtype.kind in 'iuf' and np.isfinite(arr).all(), f'Invalid geometry numeric field {key}')
    codes = classify_transitions(geometry['label'], geometry['clean_prediction'], geometry['candidate_prediction'])
    require(np.array_equal(codes, geometry['transition_code']), 'Transition partition mismatch')
    for key in ('source_index', 'candidate_index', 'transition_code', 'multiplicity'):
        require(np.array_equal(geometry[key], internal[key]), f'Internal alignment mismatch {key}')
    require(internal['trace_metrics'].shape == (n, 7, 6), 'Trace shape mismatch')
    require(np.isfinite(internal['trace_metrics']).all(), 'Nonfinite trace')
    require(internal['trace_metrics'].dtype == np.float32, 'Trace dtype mismatch')
    require(internal['trace_provenance'].shape == (n,), 'Trace provenance shape mismatch')
    require(np.count_nonzero(internal['trace_provenance'] == 1) == 51560 and np.count_nonzero(internal['trace_provenance'] == 2) == 673510,
            'Trace provenance count mismatch')
    require(internal['stage_names'].tolist() == config['stage_names'], 'Stage order mismatch')
    require(internal['metric_names'].tolist() == config['metric_names'], 'Metric order mismatch')
    changed = codes != 0
    require(int(changed.sum()) == 25820, 'Changed count mismatch')
    for key in ('source_index', 'candidate_index'):
        require(np.array_equal(internal['patch_' + key], geometry[key][changed]), f'Patch alignment mismatch {key}')
    require(internal['patch_scores'].shape == (25820, 7, 35), 'Patch score shape mismatch')
    require(internal['patch_prediction'].shape == (25820, 7), 'Patch prediction shape mismatch')
    require(np.isfinite(internal['patch_scores']).all(), 'Nonfinite patch scores')
    require(np.array_equal(internal['patch_scores'].argmax(axis=2), internal['patch_prediction']), 'Patch argmax mismatch')
    for key in ('patch_clean_class_margin', 'patch_true_class_margin'):
        require(internal[key].shape == (25820, 7) and np.isfinite(internal[key]).all(), 'Invalid patch margin')
    for key, targets in [('patch_clean_class_margin', geometry['clean_prediction'][changed]), ('patch_true_class_margin', geometry['label'][changed])]:
        a = internal['patch_scores'].copy()
        target_score = np.take_along_axis(a, targets[:, None, None].repeat(7, axis=1), axis=2).squeeze(2)
        np.put_along_axis(a, targets[:, None, None].repeat(7, axis=1), -np.inf, axis=2)
        require(np.array_equal(target_score - a.max(axis=2), internal[key]), 'Patch margin reconstruction mismatch ' + key)
    patched_clean = np.stack([clean[int(s)] for s in internal['patch_source_index']])
    for stage in (0, 3, 6):
        require(np.array_equal(internal['patch_scores'][:, stage], patched_clean), 'Positive-control scores not exactly clean')
    # Every candidate specification, source label, horizon and multiplicity is re-derived from the panel inputs.
    source_rows = []
    offset = 0
    for panel_row, stored in zip(panel, stored_source_rows):
        sid = panel_row['source_index']; x = validate_input(inputs[f'source_{sid:05d}'])
        require(x.dtype == np.uint16, 'Panel dtype mismatch')
        require(sid == stored['source_index'], 'Panel and source table order mismatch')
        digest = hashlib.sha256(x.astype('<u2').tobytes(order='C')).hexdigest()
        require(digest == panel_row['input_array_sha256'], 'Panel array hash mismatch')
        require(x.shape == (panel_row['horizon_steps'], 140) and int(x.sum()) == panel_row['input_count'], 'Panel cardinality mismatch')
        count = panel_row['unique_candidates']; sl = slice(offset, offset + count)
        require(np.all(geometry['source_index'][sl] == sid), 'Source segment mismatch')
        require(np.all(geometry['audit_rank'][sl] == panel_row['audit_rank']), 'Audit rank mismatch')
        require(np.array_equal(geometry['candidate_index'][sl], np.arange(count)), 'Missing or reordered candidate')
        require(np.all(geometry['label'][sl] == panel_row['label']), 'Source label mismatch')
        require(np.all(geometry['clean_prediction'][sl] == clean[sid].argmax()), 'Clean prediction mismatch')
        specs = enumerate_moves(x)
        for value, key in zip(specs, ('from_bin', 'to_bin', 'feature', 'multiplicity')):
            require(np.array_equal(value, geometry[key][sl]), 'Candidate operator mismatch ' + key)
        require(np.all(geometry['direction'][sl] == specs[1] - specs[0]), 'Direction mismatch')
        require(np.array_equal((specs[0] / max(1, len(x) - 1)).astype(np.float32), geometry['time_normalized'][sl]), 'Time coordinate mismatch')
        require(np.array_equal((specs[2] / 139).astype(np.float32), geometry['feature_normalized'][sl]), 'Feature coordinate mismatch')
        require(abs(float(np.sort(clean[sid])[-1] - np.sort(clean[sid])[-2]) - stored['clean_margin']) < 1e-12, 'Clean margin mismatch')
        a = {'label': np.array(panel_row['label']), 'clean_prediction': np.array(clean[sid].argmax()),
             'predictions': geometry['candidate_prediction'][sl], 'from_t': specs[0], 'to_t': specs[1], 'feature': specs[2], 'multiplicity': specs[3],
             'candidate_clean_class_margin': geometry['candidate_clean_class_margin'][sl], 'candidate_true_class_margin': geometry['candidate_true_class_margin'][sl]}
        row = derive_source_geometry(stored, a)
        errors = compare_tree(row, stored, atol=1e-12)
        require(not errors, 'Recomputed source summary mismatch: ' + str(errors[:3]))
        source_rows.append(row); offset += count
    require(offset == n, 'Candidate coverage mismatch')
    return dict(config=config, geometry=geometry, internal=internal, inputs=inputs, clean_scores=clean, panel=panel, source_rows=source_rows)


# ---------------------------------------------------------------- Table 3 and §4.2 source statistics
def total_counts(geometry: dict) -> dict:
    codes = geometry['transition_code']; weights = geometry['multiplicity'].astype(np.int64)
    result = {'unique_candidates': len(codes), 'event_weighted_candidates': int(weights.sum())}
    for code, name in enumerate(GROUPS):
        selected = codes == code
        result[name + '_unique'] = int(selected.sum())
        result[name + '_event_weighted'] = int(weights[selected].sum())
    result['class_changed_unique'] = int((codes != 0).sum())
    result['class_changed_event_weighted'] = int(weights[codes != 0].sum())
    return result


def panel_statistics(rows: list[dict], config: dict) -> dict:
    clean = [row for row in rows if row['clean_correct']]
    margins = np.array([row['clean_margin'] for row in clean], dtype=np.float64)
    rates = np.array([row['adverse_unique'] / row['unique_candidates'] for row in clean], dtype=np.float64)
    labels = np.array([row['label'] for row in clean])
    groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    rho = spearman_rho(margins, rates); reps = config['repetitions']
    rng = np.random.default_rng(config['margin_permutation_seed']); extreme = 0
    for _ in range(reps):
        permuted = rates.copy()
        for group in groups: permuted[group] = rates[rng.permutation(group)]
        extreme += spearman_rho(margins, permuted) <= rho
    rng = np.random.default_rng(config['adverse_bootstrap_seed']); boot = []
    for _ in range(reps):
        sampled = []
        for group in groups: sampled.extend(rates[rng.choice(group, size=len(group), replace=True)].tolist())
        boot.append(float(np.mean(sampled)))
    positive = sorted((row for row in clean if row['adverse_unique'] > 0), key=lambda r: (-r['adverse_unique'], r['source_index']))
    cumulative = np.cumsum([row['adverse_unique'] for row in positive]) / sum(row['adverse_unique'] for row in positive)
    horizon_rank = {}
    for name, key in (('raw_margin', lambda r: r['clean_margin']), ('margin_per_horizon', lambda r: r['clean_margin'] / r['horizon_steps'])):
        order = sorted(clean, key=lambda r: (key(r), r['source_index']))
        horizon_rank[name] = {r['source_index']: i + 1 for i, r in enumerate(order)}
    return dict(clean_correct_sources=len(clean), adverse_sensitive_sources=len(positive),
                any_change_sources=sum(row['class_changed_unique'] > 0 for row in rows),
                corrective_sources=sum(row['corrective_unique'] > 0 for row in rows),
                lateral_sources=sum(row['lateral_unique'] > 0 for row in rows),
                adverse_source_incidence=len(positive) / len(clean),
                mean_source_class_change_rate=float(np.mean([row['class_change_rate'] for row in rows])),
                mean_adverse_rate=float(rates.mean()), adverse_rate_ci95=np.percentile(boot, [2.5, 97.5]).tolist(),
                spearman_rho=rho, within_label_one_sided_p=float((extreme + 1) / (reps + 1)),
                ranked_adverse_sources=[row['source_index'] for row in positive],
                ranked_adverse_counts=[row['adverse_unique'] for row in positive],
                cumulative_adverse_share=cumulative.tolist(),
                lowest_quartile_members_raw=sorted(s for s, r in horizon_rank['raw_margin'].items() if r <= 21),
                lowest_quartile_members_per_horizon=sorted(s for s, r in horizon_rank['margin_per_horizon'].items() if r <= 21),
                source_1627_rank_raw=horizon_rank['raw_margin'].get(1627), source_1627_rank_per_horizon=horizon_rank['margin_per_horizon'].get(1627))


def accuracy_decomposition(geometry: dict, rows: list[dict]) -> dict:
    """§4.3: clean versus expected neighboring accuracy under equal-source and unique-candidate weighting."""
    correct = geometry['candidate_prediction'] == geometry['label']
    per_source = []
    for row in rows:
        selected = geometry['source_index'] == row['source_index']
        per_source.append(float(correct[selected].mean()))
    weights = np.array([row['unique_candidates'] for row in rows], dtype=np.float64)
    clean_flags = np.array([row['clean_correct'] for row in rows], dtype=np.float64)
    return dict(equal_source_clean_accuracy=float(clean_flags.mean()),
                equal_source_neighbor_accuracy=float(np.mean(per_source)),
                unique_candidate_clean_accuracy=float((clean_flags * weights).sum() / weights.sum()),
                unique_candidate_neighbor_accuracy=float(correct.mean()),
                unique_candidate_gain_percentage_points=100 * (float(correct.mean()) - float((clean_flags * weights).sum() / weights.sum())),
                corrective_minus_adverse=int((geometry['transition_code'] == 2).sum() - (geometry['transition_code'] == 1).sum()))


# ---------------------------------------------------------------- Tables 5 and 7 (final-block relative L2 change)
def final_block_descriptives(bundle: dict) -> dict:
    """Table 5 and the §4.4 in-text summaries of block-2 relative L2 change."""
    cfg = bundle['config']; stage = cfg['stage_names'].index('block_2'); metric = cfg['metric_names'].index('relative_l2_change')
    values = bundle['internal']['trace_metrics'][:, stage, metric].astype(np.float64)
    codes = bundle['geometry']['transition_code']; sources = bundle['geometry']['source_index']
    rows = []
    for code, name in enumerate(GROUPS):
        selected = codes == code
        per_source_means = [float(values[selected & (sources == sid)].mean()) for sid in np.unique(sources[selected])]
        q1, median, q3 = np.percentile(values[selected], [25, 50, 75])
        rows.append(dict(outcome=name, candidates=int(selected.sum()), sources=len(per_source_means),
                         pooled_median=float(median), pooled_q1=float(q1), pooled_q3=float(q3),
                         pooled_mean=float(values[selected].mean()), equal_source_mean=float(np.mean(per_source_means))))
    preserved = codes == 0
    within = [float(np.median(values[preserved & (sources == sid)])) for sid in np.unique(sources)]
    q1, median, q3 = np.percentile(within, [25, 50, 75])
    return dict(table=rows,
                preserved_nonzero=int(np.count_nonzero(values[preserved] > 0)),
                preserved_exact_zero=int(np.count_nonzero(values[preserved] == 0)),
                preserved_nonzero_fraction=float(np.mean(values[preserved] > 0)),
                class_changing_all_nonzero=bool(np.all(values[~preserved] > 0)),
                within_source_median_of_preserved_medians=float(median),
                within_source_preserved_median_q1=float(q1), within_source_preserved_median_q3=float(q3))


def internal_output_correlations(bundle: dict) -> dict:
    """Table 7: median within-source Spearman correlation of block-2 relative L2 change with output changes."""
    cfg = bundle['config']; stage = cfg['stage_names'].index('block_2'); metric = cfg['metric_names'].index('relative_l2_change')
    g = bundle['geometry']
    activation = bundle['internal']['trace_metrics'][:, stage, metric].astype(np.float64)
    clean_margin = np.empty(len(activation))
    for sid, scores in bundle['clean_scores'].items():
        label = int(g['label'][g['source_index'] == sid][0])
        other = scores.astype(np.float64).copy(); other[label] = -np.inf
        clean_margin[g['source_index'] == sid] = float(scores[label]) - float(other.max())
    signed = g['candidate_true_class_margin'].astype(np.float64) - clean_margin
    targets = {'score_l2': g['score_l2_change'].astype(np.float64), 'absolute_margin': np.abs(signed), 'signed_margin': signed}
    nonzero = activation > 0; preserved = g['transition_code'] == 0
    subsets = {'all_neighbors': np.ones(len(activation), dtype=bool), 'nonzero_internal_response': nonzero,
               'preserved_nonzero_response': preserved & nonzero}
    rows = []
    for subset_name, mask in subsets.items():
        row = dict(candidates=subset_name)
        for target_name, target in targets.items():
            per_source = []
            for sid in np.unique(g['source_index']):
                selected = mask & (g['source_index'] == sid)
                require(int(selected.sum()) >= 2, 'A source has fewer than two candidates in a Table 7 subset')
                per_source.append(spearman_rho(activation[selected], target[selected]))
            row[target_name] = float(np.median(per_source)); row['sources'] = len(per_source)
        rows.append(row)
    return dict(table=rows, excluded_exact_zero_candidates=int(np.count_nonzero(~nonzero)))


# ---------------------------------------------------------------- independent second route
def independent_checks(bundle: dict, internal_stats: dict, panel_stats: dict) -> dict:
    """A second counting/statistics route, not a call back into the primary helpers."""
    from scipy.stats import spearmanr, false_discovery_control
    g = bundle['geometry']; counts = Counter(); weighted = Counter()
    for label, clean, candidate, multiplicity in zip(g['label'].tolist(), g['clean_prediction'].tolist(), g['candidate_prediction'].tolist(), g['multiplicity'].tolist()):
        if candidate == clean: kind = 'class_preserved'
        elif clean == label: kind = 'adverse'
        elif candidate == label: kind = 'corrective'
        else: kind = 'lateral'
        counts[kind] += 1; weighted[kind] += multiplicity
    totals = total_counts(g)
    require(all(counts[k] == totals[k + '_unique'] and weighted[k] == totals[k + '_event_weighted'] for k in GROUPS), 'Independent scalar partition mismatch')
    correct = [row for row in bundle['source_rows'] if row['clean_correct']]
    independent_rho = float(spearmanr([row['clean_margin'] for row in correct], [row['adverse_rate'] for row in correct]).statistic)
    require(abs(independent_rho - panel_stats['spearman_rho']) < 1e-12, 'Independent Spearman mismatch')
    inf = internal_stats['source_level_inference']; families = []
    families.append(([inf['transition_vs_preserved'][group][stage] for group in GROUPS[1:] for stage in bundle['config']['stage_names']], 'fdr_bh_q_within_transition_vs_preserved_family'))
    families.append(([inf['global_vs_local'][group][f'block_{b}']['attention_minus_local_trace_divergence'] for group in GROUPS for b in (1, 2)], 'fdr_bh_q_within_global_vs_local_trace_family'))
    families.append(([inf['global_vs_local'][group][f'block_{b}']['attention_minus_local_clean_prediction_restoration'] for group in GROUPS[1:] for b in (1, 2)], 'fdr_bh_q_within_global_vs_local_patch_family'))
    for family, key in families:
        expected = false_discovery_control([row['two_sided_signflip_p'] for row in family], method='bh')
        require(np.allclose(expected, [row[key] for row in family], rtol=0, atol=1e-12), 'Independent FDR mismatch')
    return {'status': 'PASS', 'scalar_candidate_count': sum(counts.values()), 'unique_partition': dict(counts), 'weighted_partition': dict(weighted),
            'scipy_spearman_rho': independent_rho, 'fdr_families_checked': [len(v[0]) for v in families]}


# ---------------------------------------------------------------- manuscript tables and figure data
def export_tables(out: Path, bundle: dict, totals: dict, panel: dict, internal_stats: dict, final_block: dict, correlations: dict) -> None:
    outcome = []
    for name in [GROUPS[0], 'class_changed', *GROUPS[1:]]:
        outcome.append(dict(outcome=OUTCOME_LABELS[name], unique_n=totals[name + '_unique'],
                            unique_percent=100 * totals[name + '_unique'] / totals['unique_candidates'],
                            event_weighted_n=totals[name + '_event_weighted'],
                            event_weighted_percent=100 * totals[name + '_event_weighted'] / totals['event_weighted_candidates']))
    write_csv(out / 'table_03.csv', outcome)
    write_csv(out / 'table_05.csv', [dict(outcome=OUTCOME_LABELS[r['outcome']], candidates=r['candidates'], sources=r['sources'],
                                          pooled_median=r['pooled_median'], q1=r['pooled_q1'], q3=r['pooled_q3'],
                                          equal_source_mean=r['equal_source_mean']) for r in final_block['table']])
    inf = internal_stats['source_level_inference']; traces = []; patches = []
    for stage in bundle['config']['stage_names']:
        row = dict(boundary=BOUNDARY_LABELS[stage])
        for group in GROUPS[1:]:
            q = inf['transition_vs_preserved'][group][stage]
            row[f'{group}_mean_contrast'] = q['mean_contrast']; row[f'{group}_q'] = q['fdr_bh_q_within_transition_vs_preserved_family']
            row[f'{group}_ci_low'] = q['source_bootstrap95'][0]; row[f'{group}_ci_high'] = q['source_bootstrap95'][1]
            row[f'{group}_p'] = q['two_sided_signflip_p']; row[f'{group}_n_sources'] = q['n_sources']
        traces.append(row)
        if stage in ('attention_1', 'local_1', 'attention_2', 'local_2'):
            p = inf['patch_restoration']['adverse'][stage]
            patches.append(dict(boundary=BOUNDARY_LABELS[stage],
                                candidate_pooled_percent=100 * internal_stats['clean_activation_patch']['adverse'][stage]['restored_clean_prediction_rate_unique'],
                                equal_source_mean_percent=100 * p['mean_restored_clean_prediction_rate'],
                                ci95_low_percent=100 * p['source_bootstrap95_restored_clean_prediction_rate'][0],
                                ci95_high_percent=100 * p['source_bootstrap95_restored_clean_prediction_rate'][1]))
    write_csv(out / 'table_06.csv', traces)
    write_csv(out / 'table_07.csv', [dict(candidates={'all_neighbors': 'All neighbors', 'nonzero_internal_response': 'Nonzero internal response',
                                                      'preserved_nonzero_response': 'Preserved, nonzero response'}[r['candidates']],
                                          score_l2=r['score_l2'], absolute_margin=r['absolute_margin'], signed_margin=r['signed_margin'], sources=r['sources'])
                                     for r in correlations['table']])
    write_csv(out / 'table_08.csv', patches)
    write_csv(out / 'sources.csv', bundle['source_rows'])
    # Figure data: Fig. 2 = Table 3 shares; Fig. 3A = margin vs adverse rate; Fig. 3B = cumulative adverse load;
    # Fig. 4A = Table 6 with intervals; Fig. 4B = Table 8.
    changed_unique = totals['class_changed_unique']; changed_weighted = totals['class_changed_event_weighted']
    write_csv(out / 'figure_02.csv', [dict(outcome=OUTCOME_LABELS[name], share_of_all_candidates_unique_percent=100 * totals[name + '_unique'] / totals['unique_candidates'],
                                           share_of_all_candidates_event_weighted_percent=100 * totals[name + '_event_weighted'] / totals['event_weighted_candidates'],
                                           share_of_class_changes_unique_percent=100 * totals[name + '_unique'] / changed_unique if name != 'class_preserved' else '',
                                           share_of_class_changes_event_weighted_percent=100 * totals[name + '_event_weighted'] / changed_weighted if name != 'class_preserved' else '')
                                      for name in GROUPS])
    write_csv(out / 'figure_03_a.csv', [dict(source_index=r['source_index'], clean_margin=r['clean_margin'], horizon_steps=r['horizon_steps'],
                                             adverse_rate=r['adverse_rate'], adverse_unique=r['adverse_unique'], adverse_sensitive=r['adverse_unique'] > 0)
                                        for r in bundle['source_rows'] if r['clean_correct']])
    write_csv(out / 'figure_03_b.csv', [dict(rank=i + 1, source_index=s, adverse_unique=c, cumulative_percent=100 * v)
                                        for i, (s, c, v) in enumerate(zip(panel['ranked_adverse_sources'], panel['ranked_adverse_counts'], panel['cumulative_adverse_share']))])
    write_csv(out / 'figure_04_a.csv', [dict(boundary=BOUNDARY_LABELS[stage], transition=group, mean_contrast=inf['transition_vs_preserved'][group][stage]['mean_contrast'],
                                             ci_low=inf['transition_vs_preserved'][group][stage]['source_bootstrap95'][0],
                                             ci_high=inf['transition_vs_preserved'][group][stage]['source_bootstrap95'][1],
                                             q=inf['transition_vs_preserved'][group][stage]['fdr_bh_q_within_transition_vs_preserved_family'])
                                        for stage in bundle['config']['stage_names'] for group in GROUPS[1:]])
    write_csv(out / 'figure_04_b.csv', patches)


def restoration_ranges(internal_stats: dict) -> dict:
    """§4.6 in-text ranges over the four branch boundaries for corrective and lateral candidates."""
    inf = internal_stats['source_level_inference']['patch_restoration']; result = {}
    for group in ('corrective', 'lateral'):
        clean = [inf[group][s]['mean_restored_clean_prediction_rate'] for s in ('attention_1', 'local_1', 'attention_2', 'local_2')]
        truth = [inf[group][s]['mean_restored_ground_truth_rate'] for s in ('attention_1', 'local_1', 'attention_2', 'local_2')]
        result[group] = dict(clean_prediction_restoration_min=min(clean), clean_prediction_restoration_max=max(clean),
                             ground_truth_restoration_min=min(truth), ground_truth_restoration_max=max(truth))
    return result


def run(root: Path, out: Path, progress=print) -> dict:
    """Regenerate the census statistics, compare them with the reference files, write tables 3 and 5–8."""
    progress('[1/4] Checking every panel input, candidate record and replacement output', flush=True)
    b = load_and_validate(root)
    benchmark_ref = read_json(root / paths.REFERENCE['benchmark']); benchmarks = {}
    for name, relative in paths.BENCHMARK.items():
        a = load_npz(root / relative)
        m = benchmark_metrics(a['labels'], a['predictions']); benchmarks[name] = m
        require(np.array_equal(m['confusion'], a['confusion']), 'Benchmark stored confusion mismatch')
        require(np.array_equal(a['source_indices'], np.arange(len(a['labels']))), 'Benchmark order mismatch')
        if 'scores' in a: require(np.array_equal(a['scores'].argmax(axis=1), a['predictions']), 'Benchmark score argmax mismatch')
        expected = {k: benchmark_ref[name][k] for k in m if k in benchmark_ref[name]}
        require(not compare_tree({k: m[k] for k in expected}, expected), 'Benchmark metrics mismatch')
        require(m['correct'] == {'validation': 8617, 'test': 17247}[name], 'Benchmark correct-count mismatch')
    totals = total_counts(b['geometry'])
    require(not compare_tree(totals, read_json(root / paths.REFERENCE['neighborhood'])), 'Neighborhood totals mismatch')
    _, geometry_stats = geometry_aggregates(b['geometry'], b['source_rows'])
    errors = compare_tree(geometry_stats, read_json(root / paths.REFERENCE['geometry']))
    require(not errors, 'Geometry statistics mismatch ' + str(errors[:3]))
    progress('[2/4] Recomputing source bootstraps, permutations and internal contrasts', flush=True)
    panel = panel_statistics(b['source_rows'], b['config'])
    internal_stats = recompute_internal(b['geometry'], b['internal'], b['clean_scores'])
    agreement = compare_internal_statistics(internal_stats, read_json(root / paths.REFERENCE['internal']))
    write_json(out / 'census_numerical_agreement.json', agreement)
    require(agreement['status'] == 'PASS', 'Internal statistics mismatch ' + str(agreement['errors'][:5]))
    progress('[3/4] Running independent scalar counts, Spearman and FDR checks', flush=True)
    second = independent_checks(b, internal_stats, panel)
    final_block = final_block_descriptives(b)
    correlations = internal_output_correlations(b)
    accuracy = accuracy_decomposition(b['geometry'], b['source_rows'])
    require(final_block['preserved_nonzero'] == 617941, 'Preserved internal-change count mismatch')
    require(panel['clean_correct_sources'] == 84 and panel['adverse_sensitive_sources'] == 13 and panel['any_change_sources'] == 25, 'Source incidence mismatch')
    progress('[4/4] Writing tables 3, 5, 6, 7, 8 and figure data', flush=True)
    export_tables(out, b, totals, panel, internal_stats, final_block, correlations)
    report = {'status': 'PASS', 'scope': 'Statistics regenerated from released arrays, not new neural-network inference.',
              'candidates_checked': 725070, 'source_records_checked': 100, 'trace_metric_values_checked': int(b['internal']['trace_metrics'].size),
              'replacement_outputs_checked': 25820 * 7,
              'reference_tolerances': {'counts_and_predictions': 'exact', 'p_values_q_values_rates_float64': 1e-12, 'float32_derived_means_and_intervals': 1e-6},
              'max_internal_summary_absolute_difference': agreement['max_absolute_difference'],
              'benchmark': {name: {k: v for k, v in m.items() if k in ('samples', 'correct', 'accuracy', 'balanced_accuracy', 'macro_f1', 'worst_class_recall')} for name, m in benchmarks.items()},
              'outcome_partition': totals, 'panel': panel, 'accuracy_decomposition': accuracy, 'geometry': geometry_stats,
              'final_block': final_block, 'internal_output_correlations': correlations, 'restoration_ranges': restoration_ranges(internal_stats),
              'internal': internal_stats,
              'independent_checks': second, 'new_model_inference_performed': False, 'new_training_performed': False, 'raw_official_test_opened': False}
    write_json(out / 'census.json', report)
    write_json(out / 'census_benchmark_classes.json', benchmarks)
    return report
