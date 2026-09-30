"""Recompute paired execution controls from retained outputs, without inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .io import compare_tree, load_npz, read_json, sha256_file, write_csv, write_json


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def score_predictions(scores: np.ndarray, rows: int, name: str) -> np.ndarray:
    require(scores.shape == (rows, 35), f'{name}: expected [{rows},35] scores')
    require(np.issubdtype(scores.dtype, np.floating) and bool(np.isfinite(scores).all()),
            f'{name}: nonfinite or nonfloating scores')
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
    # Preserve the recorded float32 subtraction, then summarize in float64.
    delta = np.max(np.abs(a - b), axis=1).astype(np.float64)
    return dict(samples=len(delta), exact_zero=int(np.sum(delta == 0)),
                nonzero=int(np.sum(delta != 0)), median=float(np.median(delta)),
                p95=float(np.quantile(delta, .95)), p99=float(np.quantile(delta, .99)),
                maximum=float(delta.max()),
                above_1e_5=int(np.sum(delta > 1e-5)),
                above_1e_4=int(np.sum(delta > 1e-4)),
                above_1e_3=int(np.sum(delta > 1e-3)))


def contract(folder: Path, name: str) -> str:
    record = read_json(folder / name)
    digest = hashlib.sha256(json.dumps(record['payload'], sort_keys=True,
                                      separators=(',', ':')).encode()).hexdigest()
    require(digest == record['contract_sha256'], f'{name}: logical contract hash mismatch')
    return digest


def validation_alignment(gpu: dict, cpu: dict, isolated: dict) -> None:
    """Reject source reordering, input substitution and incompatible array shapes."""
    for name, arrays in [('GPU', gpu), ('CPU', cpu), ('q/k', isolated)]:
        require(np.array_equal(arrays['source_index'], np.arange(9981)),
                f'{name}: validation source order changed')
        require(arrays['label'].shape == (9981,), f'{name}: validation label shape changed')
        require(np.array_equal(arrays['label'], gpu['label']), f'{name}: labels differ')
    require(np.array_equal(cpu['horizon'], gpu['native_horizon']), 'Native horizons differ')
    require(np.array_equal(cpu['input_sha256'], gpu['input_sha256']), 'Transformed inputs differ')
    require(bool(np.all(gpu['padded_horizon'] >= gpu['native_horizon'])), 'Negative padding')


def analyze(root: Path, progress=print) -> dict:
    folder = root / 'data/execution_controls'
    progress('[1/5] Loading retained validation and candidate outputs', flush=True)
    gpu = load_npz(folder / 'gpu_validation_predictions.npz')
    cpu = load_npz(folder / 'cpu_validation_predictions.npz')
    isolated = load_npz(folder / 'qk_isolation_validation.npz')
    paired = load_npz(folder / 'paired_candidate_predictions.npz')
    history_hash = contract(folder, 'gpu_execution_contract.json')
    mechanism_hash = contract(folder, 'mechanism_device_contract.json')
    require(str(gpu['contract_sha256']) == history_hash, 'GPU output contract differs')
    for name, arrays in [('q/k', isolated), ('candidate', paired)]:
        require(str(arrays['contract_sha256']) == mechanism_hash, f'{name} output contract differs')
    payload = read_json(folder / 'mechanism_device_contract.json')['payload']
    require(payload['official_test_access'] is False, 'Unexpected official-test access contract')
    for key, path in [('history_predictions_sha256', folder / 'gpu_validation_predictions.npz'),
                      ('checkpoint_sha256', root / 'data/model/frozen_checkpoint.pt'),
                      ('geometry_sha256', root / 'data/neighborhood/candidate_records.npz'),
                      ('script_sha256', root / 'scripts/recorded/mechanism_device_controls_20260930.py')]:
        expected = payload[key] if key == 'script_sha256' else payload['provenance'][key]
        require(sha256_file(path) == expected, f'{key}: provenance hash mismatch')
    validation_alignment(gpu, cpu, isolated)

    progress('[2/5] Checking labels, inputs and the canonical singleton panel', flush=True)
    conditions = {'CPU_singleton': cpu['scores'], 'GPU_singleton': gpu['B1_native_scores'],
                  'GPU_padding_matched_singleton': gpu['B1_padding_matched_scores'],
                  'GPU_batch256': gpu['B256_scores'],
                  'GPU_batch256_reversed': gpu['B256_reversed_scores'],
                  'GPU_qk_isolated_batch256': isolated['qk_isolated_scores'],
                  'GPU_qk_isolated_batch256_reversed': isolated['qk_isolated_reversed_scores']}
    prediction = {name: score_predictions(scores, 9981, name) for name, scores in conditions.items()}
    for key in ('B1_native', 'B1_padding_matched', 'B256', 'B256_reversed'):
        require(np.array_equal(gpu[f'{key}_scores'].argmax(axis=1), gpu[f'{key}_prediction']),
                f'{key}: saved prediction disagrees with scores')
    require(np.array_equal(cpu['scores'].argmax(axis=1), cpu['prediction']), 'CPU saved predictions differ')
    require(np.array_equal(isolated['original_scores'], gpu['B256_scores']), 'Historical batch scores differ')
    require(np.array_equal(isolated['original_reversed_scores'], gpu['B256_reversed_scores']),
            'Historical reversed batch scores differ')
    benchmark = load_npz(root / 'data/benchmark/validation_predictions.npz')
    require(np.array_equal(benchmark['predictions'], prediction['GPU_batch256']),
            'Archived validation benchmark labels differ')
    clean = load_npz(root / 'data/panel/clean_scores.npz')
    panel_scores = gpu['B1_native_scores'][clean['source_index']]
    panel = dict(sources=len(clean['source_index']),
                 score_vectors_bitwise_equal=bool(np.array_equal(panel_scores, clean['scores'])),
                 maximum_score_difference=float(np.max(np.abs(panel_scores - clean['scores']))))
    require(panel['score_vectors_bitwise_equal'], 'Canonical panel singleton scores differ')
    metrics = {name: dict(correct=int(np.sum(pred == gpu['label'])), samples=len(pred),
                          accuracy=float(np.mean(pred == gpu['label'])))
               for name, pred in prediction.items()}
    pairs = [('CPU_singleton', 'GPU_singleton'),
             ('GPU_singleton', 'GPU_padding_matched_singleton'),
             ('GPU_singleton', 'GPU_batch256'),
             ('GPU_padding_matched_singleton', 'GPU_batch256'),
             ('GPU_batch256', 'GPU_batch256_reversed'),
             ('GPU_qk_isolated_batch256', 'GPU_qk_isolated_batch256_reversed'),
             ('GPU_padding_matched_singleton', 'GPU_qk_isolated_batch256')]
    contrasts = []
    for a, b in pairs:
        differences = prediction[a] != prediction[b]
        contrasts.append(dict(a=a, b=b, changed_labels=int(differences.sum()),
                              changed_source_ids=gpu['source_index'][differences].tolist(),
                              score_difference=score_difference(conditions[a], conditions[b])))

    progress('[3/5] Recomputing padding-gap counts and q/k isolation residuals', flush=True)
    gap = gpu['padded_horizon'] - gpu['native_horizon']
    changed = prediction['GPU_singleton'] != prediction['GPU_padding_matched_singleton']
    require(not bool(np.any(changed & (gap == 0))), 'A source without padding changes its label')
    bins = []
    for lo, hi in [(0, 0), (1, 5), (6, 10), (11, 20), (21, 40), (41, 80), (81, 100000)]:
        selected = (gap >= lo) & (gap <= hi)
        bins.append(dict(padding_bins_min=lo, padding_bins_max=hi, sources=int(selected.sum()),
                         changed_labels=int(np.sum(changed & selected)),
                         changed_fraction=float(np.mean(changed[selected]))))

    progress('[4/5] Aligning all 25,820 replays with the original outcome map', flush=True)
    geometry = load_npz(root / 'data/neighborhood/candidate_records.npz')
    rows = np.flatnonzero(geometry['transition_code'] != 0)
    require(np.array_equal(paired['geometry_row_index'], rows), 'Candidate cohort or row order differs')
    for original, replay in [('source_index', 'source_index'), ('candidate_index', 'candidate_index'),
                             ('label', 'label'), ('from_bin', 'from_bin'), ('to_bin', 'to_bin'),
                             ('feature', 'feature'), ('candidate_prediction', 'canonical_candidate_prediction'),
                             ('transition_code', 'canonical_transition_code')]:
        require(np.array_equal(geometry[original][rows], paired[replay]), f'Candidate {original} differs')
    device_predictions, codes = {}, {}
    for device, clean_name in [('CPU', 'CPU_singleton'), ('GPU', 'GPU_singleton')]:
        pred = score_predictions(paired[f'{device}_scores'], len(rows), f'{device} candidates')
        require(np.array_equal(pred, paired[f'{device}_prediction']), f'{device} candidate predictions differ')
        require(np.array_equal(paired[f'{device}_clean_prediction'],
                               prediction[clean_name][paired['source_index']]),
                f'{device} clean/candidate source pairing differs')
        code = transition(paired['label'], paired[f'{device}_clean_prediction'], pred)
        require(np.array_equal(code, paired[f'{device}_transition_code']), f'{device} transition codes differ')
        device_predictions[device], codes[device] = pred, code
    canonical = paired['canonical_transition_code']
    source_sets = {device: np.unique(paired['source_index'][code == 1]).tolist()
                   for device, code in codes.items()}
    canonical_sources = np.unique(paired['source_index'][canonical == 1]).tolist()
    witnesses = []
    for sid in canonical_sources:
        j = np.flatnonzero((paired['source_index'] == sid) & (canonical == 1))[0]
        witnesses.append(dict(source_index=sid, candidate_index=int(paired['candidate_index'][j]),
                              CPU_is_adverse=bool(codes['CPU'][j] == 1),
                              GPU_is_adverse=bool(codes['GPU'][j] == 1)))
    by_type = {}
    for code, name in [(1, 'adverse'), (2, 'corrective'), (3, 'lateral')]:
        selected = canonical == code
        by_type[name] = dict(candidates=int(selected.sum()),
                             CPU_GPU_label_agreement=float(np.mean(device_predictions['CPU'][selected] ==
                                                                    device_predictions['GPU'][selected])),
                             CPU_retains_canonical_label=float(np.mean(device_predictions['CPU'][selected] ==
                                                                       paired['canonical_candidate_prediction'][selected])),
                             GPU_retains_canonical_label=float(np.mean(device_predictions['GPU'][selected] ==
                                                                       paired['canonical_candidate_prediction'][selected])),
                             CPU_retains_canonical_type=float(np.mean(codes['CPU'][selected] == code)),
                             GPU_retains_canonical_type=float(np.mean(codes['GPU'][selected] == code)))
    device_report = dict(candidates=len(rows),
                         CPU_GPU_label_agreements=int(np.sum(device_predictions['CPU'] == device_predictions['GPU'])),
                         CPU_GPU_transition_agreements=int(np.sum(codes['CPU'] == codes['GPU'])),
                         score_difference=score_difference(paired['CPU_scores'], paired['GPU_scores']),
                         canonical_adverse_sources=canonical_sources, adverse_sources=source_sets,
                         first_canonical_adverse_witnesses=witnesses, by_canonical_transition=by_type,
                         originally_preserved_candidates_replayed=False)
    progress('[5/5] Finished retained-output recomputation; no model inference executed', flush=True)
    return dict(schema_version=1, validation_sources=9981, canonical_panel=panel,
                contracts=dict(gpu_execution=history_hash, mechanism_device=mechanism_hash),
                conditions=metrics, paired_validation_contrasts=contrasts, padding_gap_counts=bins,
                class_changing_device_replay=device_report, new_model_inference=False,
                official_test_access=False)


def reproduce(root: Path, out: Path) -> dict:
    report = analyze(root)
    expected = read_json(root / 'data/reference/execution_control_statistics.json')
    differences = compare_tree(report, expected, atol=1e-12)
    write_json(out / 'execution_control_statistics.json', report)
    write_json(out / 'numerical_agreement.json', dict(status='FAIL' if differences else 'PASS',
                                                   differences=differences))
    write_csv(out / 'validation_conditions.csv', [dict(condition=name, **value)
                                                 for name, value in report['conditions'].items()])
    write_csv(out / 'paired_validation_contrasts.csv',
              [dict(a=row['a'], b=row['b'], changed_labels=row['changed_labels'],
                    **row['score_difference']) for row in report['paired_validation_contrasts']])
    write_csv(out / 'padding_gap_counts.csv', report['padding_gap_counts'])
    require(not differences, 'Execution-control reference comparison failed: ' + '; '.join(differences[:10]))
    return dict(status='PASS', validation_sources=9981, paired_candidates=25820,
                new_model_inference=False, official_test_access=False)
