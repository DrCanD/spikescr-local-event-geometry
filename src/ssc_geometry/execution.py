"""Execution controls on the frozen checkpoint (manuscript §4.1, §4.8 and Appendix A; Tables 2, A.1–A.3).

All comparisons are recomputed from the retained score arrays of the full validation set and of the
25,820 class-changing candidates. The recorded contracts are kept verbatim because their hash is stored
inside those arrays. No model is executed.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from . import paths
from .census import benchmark_metrics
from .io import compare_tree, load_npz, read_json, sha256_file, write_csv, write_json

CONDITION_LABELS = {'CPU_singleton': 'CPU singleton', 'GPU_singleton': 'Native GPU singleton',
                    'GPU_padding_matched_singleton': 'Padding-matched GPU singleton', 'GPU_batch256': 'GPU batch 256',
                    'GPU_batch256_reversed': 'Reversed GPU batch 256', 'GPU_qk_isolated_batch256': 'q/k-isolated batch 256',
                    'GPU_qk_isolated_batch256_reversed': 'Reversed q/k-isolated batch 256', 'archived_batch256': 'Archived batch 256'}
PAIRS = [('CPU_singleton', 'GPU_singleton'), ('GPU_singleton', 'GPU_padding_matched_singleton'),
         ('GPU_padding_matched_singleton', 'GPU_batch256'), ('GPU_singleton', 'GPU_batch256'),
         ('GPU_batch256', 'GPU_batch256_reversed'), ('GPU_padding_matched_singleton', 'GPU_qk_isolated_batch256'),
         ('GPU_qk_isolated_batch256', 'GPU_qk_isolated_batch256_reversed'), ('archived_batch256', 'GPU_batch256'),
         ('GPU_batch256', 'CPU_singleton')]
PADDING_BINS = [(0, 0), (1, 5), (6, 10), (11, 20), (21, 40), (41, 80), (81, 100000)]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def score_predictions(scores: np.ndarray, rows: int, name: str) -> np.ndarray:
    require(scores.shape == (rows, 35), f'{name}: expected [{rows},35] scores')
    require(np.issubdtype(scores.dtype, np.floating) and bool(np.isfinite(scores).all()), f'{name}: nonfinite or nonfloating scores')
    return scores.argmax(axis=1)


def transition(labels: np.ndarray, clean: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    require(labels.shape == clean.shape == candidate.shape, 'Transition arrays are not aligned')
    code = np.zeros(labels.shape, dtype=np.uint8)
    changed = candidate != clean
    code[changed & (clean == labels)] = 1
    code[changed & (clean != labels) & (candidate == labels)] = 2
    code[changed & (clean != labels) & (candidate != labels)] = 3
    return code


def score_difference(a: np.ndarray, b: np.ndarray) -> dict:
    require(a.shape == b.shape and a.ndim == 2, 'Paired score arrays are not aligned')
    delta = np.max(np.abs(a - b), axis=1).astype(np.float64)  # float32 subtraction preserved, summary in float64
    return dict(samples=len(delta), exact_zero=int(np.sum(delta == 0)), nonzero=int(np.sum(delta != 0)),
                median=float(np.median(delta)), p95=float(np.quantile(delta, .95)), p99=float(np.quantile(delta, .99)),
                maximum=float(delta.max()), above_1e_5=int(np.sum(delta > 1e-5)), above_1e_4=int(np.sum(delta > 1e-4)),
                above_1e_3=int(np.sum(delta > 1e-3)))


def label_contrast(labels: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict:
    """Changed labels split into correct-to-wrong, wrong-to-correct and wrong-with-different-class."""
    changed = a != b
    return dict(changed_labels=int(changed.sum()), changed_fraction=float(changed.mean()),
                correct_to_wrong=int(np.sum(changed & (a == labels))), wrong_to_correct=int(np.sum(changed & (b == labels))),
                wrong_to_different_wrong=int(np.sum(changed & (a != labels) & (b != labels))))


def contract(root: Path, relative: str) -> tuple[str, dict]:
    record = read_json(root / relative)
    digest = hashlib.sha256(json.dumps(record['payload'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    require(digest == record['contract_sha256'], f'{relative}: contract hash mismatch')
    return digest, record['payload']


def validation_alignment(gpu: dict, cpu: dict, isolated: dict) -> None:
    """Reject source reordering, input substitution and incompatible array shapes."""
    for name, arrays in [('GPU', gpu), ('CPU', cpu), ('q/k', isolated)]:
        require(np.array_equal(arrays['source_index'], np.arange(9981)), f'{name}: validation source order changed')
        require(arrays['label'].shape == (9981,), f'{name}: validation label shape changed')
        require(np.array_equal(arrays['label'], gpu['label']), f'{name}: labels differ')
    require(np.array_equal(cpu['horizon'], gpu['native_horizon']), 'Native horizons differ')
    require(np.array_equal(cpu['input_sha256'], gpu['input_sha256']), 'Transformed inputs differ')
    require(bool(np.all(gpu['padded_horizon'] >= gpu['native_horizon'])), 'Negative padding')


def analyze(root: Path, progress=print) -> dict:
    progress('[1/5] Loading retained validation and candidate outputs', flush=True)
    gpu = load_npz(root / paths.EXECUTION_GPU_CONDITIONS)
    cpu = load_npz(root / paths.EXECUTION_CPU_SINGLETON)
    isolated = load_npz(root / paths.EXECUTION_QK_ISOLATION)
    paired = load_npz(root / paths.EXECUTION_CANDIDATE_REPLAY)
    conditions_hash, _ = contract(root, paths.CONTRACT_GPU_CONDITIONS)
    isolation_hash, payload = contract(root, paths.CONTRACT_ISOLATION_AND_REPLAY)
    runner_up_hash, _ = contract(root, paths.CONTRACT_GRADIENT_RUNNER_UP)
    require(str(gpu['contract_sha256']) == conditions_hash, 'GPU output contract differs')
    require(str(cpu['contract_sha256']) == runner_up_hash, 'CPU singleton output contract differs')
    for name, arrays in [('q/k', isolated), ('candidate', paired)]:
        require(str(arrays['contract_sha256']) == isolation_hash, f'{name} output contract differs')
    require(payload['official_test_access'] is False, 'Unexpected official-test access contract')
    records = read_json(root / paths.RECORDS)
    executed = {e['role']: e['sha256'] for e in records['executed_sources']}
    require(payload['script_sha256'] == executed['q/k isolation and paired CPU/GPU replay'], 'Executed-source hash is not the recorded one')
    for key, relative in [('history_predictions_sha256', paths.EXECUTION_GPU_CONDITIONS), ('checkpoint_sha256', paths.CHECKPOINT),
                          ('geometry_sha256', paths.CANDIDATES)]:
        require(sha256_file(root / relative) == payload['provenance'][key], f'{key}: provenance hash mismatch')
    validation_alignment(gpu, cpu, isolated)

    progress('[2/5] Comparing the eight execution conditions on all 9,981 validation sources', flush=True)
    benchmark = load_npz(root / paths.BENCHMARK['validation'])
    conditions = {'CPU_singleton': cpu['scores'], 'GPU_singleton': gpu['B1_native_scores'],
                  'GPU_padding_matched_singleton': gpu['B1_padding_matched_scores'], 'GPU_batch256': gpu['B256_scores'],
                  'GPU_batch256_reversed': gpu['B256_reversed_scores'], 'GPU_qk_isolated_batch256': isolated['qk_isolated_scores'],
                  'GPU_qk_isolated_batch256_reversed': isolated['qk_isolated_reversed_scores']}
    prediction = {name: score_predictions(scores, 9981, name) for name, scores in conditions.items()}
    prediction['archived_batch256'] = benchmark['predictions']  # archived labels only; the fresh replay keeps the scores
    for key in ('B1_native', 'B1_padding_matched', 'B256', 'B256_reversed'):
        require(np.array_equal(gpu[f'{key}_scores'].argmax(axis=1), gpu[f'{key}_prediction']), f'{key}: saved prediction disagrees with scores')
    require(np.array_equal(cpu['scores'].argmax(axis=1), cpu['prediction']), 'CPU saved predictions differ')
    require(np.array_equal(isolated['original_scores'], gpu['B256_scores']), 'Historical batch scores differ')
    require(np.array_equal(isolated['original_reversed_scores'], gpu['B256_reversed_scores']), 'Historical reversed batch scores differ')
    require(np.array_equal(benchmark['labels'], gpu['label']) and np.array_equal(benchmark['predictions'], prediction['archived_batch256']),
            'Archived validation benchmark differs')
    require(np.array_equal(benchmark['predictions'], prediction['GPU_batch256']), 'Archived validation benchmark labels differ from the fresh replay')
    clean = load_npz(root / paths.PANEL_CLEAN_SCORES)
    panel_scores = gpu['B1_native_scores'][clean['source_index']]
    panel = dict(sources=len(clean['source_index']), score_vectors_bitwise_equal=bool(np.array_equal(panel_scores, clean['scores'])),
                 maximum_score_difference=float(np.max(np.abs(panel_scores - clean['scores']))))
    require(panel['score_vectors_bitwise_equal'], 'Canonical panel singleton scores differ')
    panel_labels = gpu['label'][clean['source_index']]
    panel['correct_singleton'] = int(np.sum(prediction['GPU_singleton'][clean['source_index']] == panel_labels))
    panel['correct_batch256'] = int(np.sum(prediction['GPU_batch256'][clean['source_index']] == panel_labels))
    panel['labels_differing_singleton_vs_batch256'] = int(np.sum(prediction['GPU_singleton'][clean['source_index']] != prediction['GPU_batch256'][clean['source_index']]))
    metrics = {name: benchmark_metrics(gpu['label'], pred) for name, pred in prediction.items()}
    metrics = {name: {k: v for k, v in m.items() if k in ('samples', 'correct', 'accuracy', 'balanced_accuracy', 'macro_f1', 'worst_class_recall')}
               for name, m in metrics.items()}
    contrasts = []
    for a, b in PAIRS:
        differences = prediction[a] != prediction[b]
        contrasts.append(dict(a=a, b=b, accuracy_a=metrics[a]['accuracy'], accuracy_b=metrics[b]['accuracy'],
                              **label_contrast(gpu['label'], prediction[a], prediction[b]),
                              changed_source_ids=gpu['source_index'][differences].tolist(),
                              changed_classes={str(int(s)): [int(prediction[a][s]), int(prediction[b][s])] for s in gpu['source_index'][differences][:16]},
                              score_difference=score_difference(conditions[a], conditions[b]) if a in conditions and b in conditions else None))
    cross_shape = next(c for c in contrasts if (c['a'], c['b']) == ('GPU_padding_matched_singleton', 'GPU_qk_isolated_batch256'))
    cross_shape['sources_above_1e_4'] = cross_shape['score_difference']['above_1e_4']

    progress('[3/5] Recomputing padding-gap counts', flush=True)
    gap = gpu['padded_horizon'] - gpu['native_horizon']
    changed = prediction['GPU_singleton'] != prediction['GPU_padding_matched_singleton']
    require(not bool(np.any(changed & (gap == 0))), 'A source without padding changes its label')
    bins = []
    for lo, hi in PADDING_BINS:
        selected = (gap >= lo) & (gap <= hi)
        bins.append(dict(padding_bins_min=lo, padding_bins_max=hi, sources=int(selected.sum()), changed_labels=int(np.sum(changed & selected)),
                         changed_fraction=float(np.mean(changed[selected]))))

    progress('[4/5] Aligning all 25,820 device replays with the canonical outcome map', flush=True)
    geometry = load_npz(root / paths.CANDIDATES)
    rows = np.flatnonzero(geometry['transition_code'] != 0)
    require(np.array_equal(paired['geometry_row_index'], rows), 'Candidate cohort or row order differs')
    for original, replay in [('source_index', 'source_index'), ('candidate_index', 'candidate_index'), ('label', 'label'), ('from_bin', 'from_bin'),
                             ('to_bin', 'to_bin'), ('feature', 'feature'), ('candidate_prediction', 'canonical_candidate_prediction'),
                             ('transition_code', 'canonical_transition_code')]:
        require(np.array_equal(geometry[original][rows], paired[replay]), f'Candidate {original} differs')
    device_predictions, codes = {}, {}
    for device, clean_name in [('CPU', 'CPU_singleton'), ('GPU', 'GPU_singleton')]:
        pred = score_predictions(paired[f'{device}_scores'], len(rows), f'{device} candidates')
        require(np.array_equal(pred, paired[f'{device}_prediction']), f'{device} candidate predictions differ')
        require(np.array_equal(paired[f'{device}_clean_prediction'], prediction[clean_name][paired['source_index']]), f'{device} clean/candidate source pairing differs')
        code = transition(paired['label'], paired[f'{device}_clean_prediction'], pred)
        require(np.array_equal(code, paired[f'{device}_transition_code']), f'{device} transition codes differ')
        device_predictions[device], codes[device] = pred, code
    canonical = paired['canonical_transition_code']
    source_sets = {device: np.unique(paired['source_index'][code == 1]).tolist() for device, code in codes.items()}
    canonical_sources = np.unique(paired['source_index'][canonical == 1]).tolist()
    panel['adverse_sensitive_sources_clean_incorrect_under_batch256'] = int(np.sum(prediction['GPU_batch256'][canonical_sources] != gpu['label'][canonical_sources]))
    witnesses = []
    for sid in canonical_sources:
        j = np.flatnonzero((paired['source_index'] == sid) & (canonical == 1))[0]
        witnesses.append(dict(source_index=sid, candidate_index=int(paired['candidate_index'][j]), CPU_is_adverse=bool(codes['CPU'][j] == 1),
                              GPU_is_adverse=bool(codes['GPU'][j] == 1), CPU_same_class=bool(device_predictions['CPU'][j] == paired['canonical_candidate_prediction'][j])))
    by_type = {}
    for code, name in [(1, 'adverse'), (2, 'corrective'), (3, 'lateral')]:
        selected = canonical == code
        by_type[name] = dict(candidates=int(selected.sum()),
                             CPU_GPU_label_agreement=float(np.mean(device_predictions['CPU'][selected] == device_predictions['GPU'][selected])),
                             CPU_same_class=int(np.sum(device_predictions['CPU'][selected] == paired['canonical_candidate_prediction'][selected])),
                             GPU_same_class=int(np.sum(device_predictions['GPU'][selected] == paired['canonical_candidate_prediction'][selected])),
                             CPU_same_type=int(np.sum(codes['CPU'][selected] == code)), GPU_same_type=int(np.sum(codes['GPU'][selected] == code)),
                             CPU_returns_to_clean_class=int(np.sum(device_predictions['CPU'][selected] == paired['CPU_clean_prediction'][selected])))
    panel_clean = dict(CPU_GPU_clean_label_disagreements=int(np.sum(prediction['CPU_singleton'][clean['source_index']] != prediction['GPU_singleton'][clean['source_index']])))
    device_report = dict(candidates=len(rows), CPU_GPU_label_agreements=int(np.sum(device_predictions['CPU'] == device_predictions['GPU'])),
                         CPU_GPU_transition_agreements=int(np.sum(codes['CPU'] == codes['GPU'])),
                         CPU_same_type_total=int(sum(v['CPU_same_type'] for v in by_type.values())),
                         CPU_same_class_total=int(sum(v['CPU_same_class'] for v in by_type.values())),
                         score_difference=score_difference(paired['CPU_scores'], paired['GPU_scores']),
                         canonical_adverse_sources=canonical_sources, adverse_sources=source_sets,
                         first_canonical_adverse_witnesses=witnesses, by_canonical_transition=by_type, panel_clean_labels=panel_clean,
                         originally_preserved_candidates_replayed=False)
    recorded = read_json(root / paths.EXECUTION_SUMMARY)
    iso = recorded['isolation_and_replay']
    require(iso['D_device_replay']['CPU_GPU_prediction_disagreements'] == len(rows) - device_report['CPU_GPU_label_agreements'], 'Recorded device-replay agreement differs')
    require(iso['B_full_validation']['isolated_order_changed_labels'] == next(c['changed_labels'] for c in contrasts if c['a'] == 'GPU_qk_isolated_batch256'),
            'Recorded isolation order invariance differs')
    pd = iso['padding_decomposition']
    summary_values = dict(
        padding_decomposition=dict(diagnostic_sources=pd['diagnostic_sources'], changed_sources=pd['selected_from_historical_padding_changes'],
                                   unchanged_positive_padding_controls=pd['unchanged_positive_padding_controls'],
                                   restored_by_excluding_extra_bins=pd['new_padding_changes_restored_by_excluding_extra_bins'],
                                   prefix_only_labels_different_from_native=pd['prefix_only_labels_different_from_native'],
                                   native_plus_observed_extra_changes_label=pd['native_plus_observed_extra_changes_label'],
                                   native_and_padded_labels_reproduced=pd['historical_native_label_disagreements'] == 0 and pd['historical_padded_label_disagreements'] == 0),
        pair_controls=dict(pairs=len({p['pair'] for p in iso['B_pair_controls']['pairs']}),
                           joint_isolation_order_changed_labels=int(sum(p['order_changed_labels'] for p in iso['B_pair_controls']['pairs'] if p['mode'] == 'qk')),
                           joint_isolation_max_order_score_difference=max(p['order_score_max_abs'] for p in iso['B_pair_controls']['pairs'] if p['mode'] == 'qk'),
                           joint_isolation_max_versus_padded_singleton=max(p['versus_padded_singleton_max_abs'] for p in iso['B_pair_controls']['pairs'] if p['mode'] == 'qk'),
                           nonzero_order_difference_in_every_original_q_or_k_pair=all(p['order_score_max_abs'] > 0 for p in iso['B_pair_controls']['pairs'] if p['mode'] in ('original', 'q', 'k'))),
        identical_singleton_replays=recorded['gpu_conditions']['identical_singleton_replays'],
        note='Values read from data/execution/summary.json: the per-source arrays behind them (padding decomposition, pair controls) are not in this repository.')
    progress('[5/5] Finished; no model inference executed', flush=True)
    test = load_npz(root / paths.BENCHMARK['test'])
    test_metrics = benchmark_metrics(test['labels'], test['predictions'])
    table_02 = [dict(split='Validation, batch 256', **{k: metrics['GPU_batch256'][k] for k in ('correct', 'samples', 'accuracy', 'balanced_accuracy', 'macro_f1')}),
                dict(split='Validation, GPU singleton (audit path)', **{k: metrics['GPU_singleton'][k] for k in ('correct', 'samples', 'accuracy', 'balanced_accuracy', 'macro_f1')}),
                dict(split='Validation, CPU singleton', **{k: metrics['CPU_singleton'][k] for k in ('correct', 'samples', 'accuracy', 'balanced_accuracy', 'macro_f1')}),
                dict(split='Official test, batch 256', **{k: test_metrics[k] for k in ('correct', 'samples', 'accuracy', 'balanced_accuracy', 'macro_f1')})]
    return dict(schema_version=2, validation_sources=9981, canonical_panel=panel,
                contracts=dict(gpu_conditions=conditions_hash, isolation_and_replay=isolation_hash, gradient_runner_up=runner_up_hash),
                conditions=metrics, table_02=table_02, paired_validation_contrasts=contrasts, padding_gap_counts=bins,
                class_changing_device_replay=device_report, summary_values=summary_values, new_model_inference=False, official_test_access=False)


def run(root: Path, out: Path, progress=print) -> dict:
    report = analyze(root, progress)
    expected = read_json(root / paths.REFERENCE['execution'])
    differences = compare_tree({k: report[k] for k in expected}, expected, atol=1e-12)
    write_json(out / 'execution_numerical_agreement.json', dict(status='FAIL' if differences else 'PASS', differences=differences))
    write_csv(out / 'table_02.csv', report['table_02'])
    write_csv(out / 'table_A1.csv', [dict(condition_a=CONDITION_LABELS[c['a']], condition_b=CONDITION_LABELS[c['b']], accuracy_a_percent=100 * c['accuracy_a'],
                                          accuracy_b_percent=100 * c['accuracy_b'], changed_labels=c['changed_labels'], changed_percent=100 * c['changed_fraction'],
                                          correct_to_wrong=c['correct_to_wrong'], wrong_to_correct=c['wrong_to_correct'], wrong_to_different_wrong=c['wrong_to_different_wrong'])
                                     for c in report['paired_validation_contrasts'][:8]])
    write_csv(out / 'table_A2.csv', [dict(added_bins_min=b['padding_bins_min'], added_bins_max=b['padding_bins_max'] if b['padding_bins_max'] < 100000 else '',
                                          sources=b['sources'], changed=b['changed_labels'], rate_percent=100 * b['changed_fraction']) for b in report['padding_gap_counts']])
    by = report['class_changing_device_replay']['by_canonical_transition']
    a3 = [dict(canonical_type=name.capitalize(), candidates=by[name]['candidates'], cpu_same_type=by[name]['CPU_same_type'], cpu_same_class=by[name]['CPU_same_class'],
               gpu_same_type=by[name]['GPU_same_type'], gpu_same_class=by[name]['GPU_same_class']) for name in ('adverse', 'corrective', 'lateral')]
    a3.append(dict(canonical_type='Total', candidates=sum(r['candidates'] for r in a3), cpu_same_type=sum(r['cpu_same_type'] for r in a3),
                   cpu_same_class=sum(r['cpu_same_class'] for r in a3), gpu_same_type=sum(r['gpu_same_type'] for r in a3), gpu_same_class=sum(r['gpu_same_class'] for r in a3)))
    write_csv(out / 'table_A3.csv', a3)
    write_csv(out / 'execution_conditions.csv', [dict(condition=CONDITION_LABELS[name], **value) for name, value in report['conditions'].items()])
    write_csv(out / 'execution_score_distributions.csv', [dict(a=CONDITION_LABELS[c['a']], b=CONDITION_LABELS[c['b']], **c['score_difference']) for c in report['paired_validation_contrasts'] if c['score_difference']]
              + [dict(a='CPU candidate replay', b='GPU candidate replay', **report['class_changing_device_replay']['score_difference'])])
    require(not differences, 'Execution-control reference comparison failed: ' + '; '.join(differences[:10]))
    write_json(out / 'execution.json', dict(status='PASS', **report))
    return dict(status='PASS', **report)
