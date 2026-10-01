"""Two count-readout replicas on the same 100 utterance identities (manuscript §4.7, Table 9).

Table 9 and the paired comparisons are regenerated from the per-source outcome tables; the paired
within-label source bootstrap uses the recorded repetition count and seed. The replica weights
and raw candidate maps are not part of this repository, so no model is executed here.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from . import paths
from .io import compare_tree, load_npz, read_json, read_numeric_csv, write_csv, write_json

REPLICAS = {'A': 341969035, 'B': 554720563}
OUTCOMES = ('T0', 'T1', 'T2', 'T3', 'adverse', 'corrective', 'lateral')
ROW_LABELS = {'T0': 'T0: trace unchanged', 'T1': 'T1: trace changed, counts preserved',
              'T2': 'T2: counts changed, class preserved', 'T3': 'T3: class changed',
              'final_layer_trace_changed_counts_preserved': 'Final-layer trace changed, counts preserved',
              'adverse': 'Adverse neighbors (subset of T3)', 'initially_correct_sources': 'Initially correct sources (of 100)',
              'robust_correct_sources': 'Correct and robust sources (of 100)'}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load(root: Path) -> dict:
    """Load the replica records and check that the per-source and paired tables describe the same census."""
    source = read_numeric_csv(root / paths.REPLICA_SOURCE_OUTCOMES)
    paired = read_numeric_csv(root / paths.REPLICA_PAIRED_SOURCES)
    partition = read_numeric_csv(root / paths.REPLICA_READOUT_PARTITION)
    require(len(source) == 200 and len(paired) == 100, 'Replica panel cardinality differs')
    by_replica, validation = {}, {}
    for name, seed in REPLICAS.items():
        own = {r['source_index']: r for r in source if r['seed'] == seed}
        require(len(own) == 100, 'Replica source identities differ')
        for r in paired:
            s = own[r['source_index']]
            require(s['complete'] and s['candidates_done'] == r['neighbors'] == s['candidates'],
                    'Replica source census is incomplete')
            require(s['label'] == r['label'], 'Replica labels differ')
            require(bool(s['clean_correct']) == r[f'{name}_correct'], 'Replica clean cohort differs')
            for key in OUTCOMES:
                require(s[key] == r[f'{name}_{key}'], 'Replica paired counts differ')
                require(abs(r[f'{name}_{key}_source_rate'] - s[key] / s['candidates']) < 1e-12, 'Replica source rate differs from its count')
            require(s['T0'] + s['T1'] + s['T2'] + s['T3'] == s['candidates'], 'Replica outcome partition does not cover the neighborhood')
            require(s['adverse'] + s['corrective'] + s['lateral'] == s['T3'], 'Replica T3 partition differs')
        require(sum(s['candidates'] for s in own.values()) == 1297705, 'Replica neighborhood total differs')
        v = load_npz(root / paths.REPLICA_VALIDATION[name])
        require(np.array_equal(v['source_indices'], np.arange(9981)), 'Replica validation source order differs')
        require(np.array_equal(v['logits'].argmax(axis=1), v['predictions']), 'Replica saved validation prediction disagrees with scores')
        for sid, s in own.items():
            require(int(v['predictions'][sid]) == s['clean_prediction'] and int(v['labels'][sid]) == s['label'],
                    'Replica panel prediction differs from its full-validation run')
        by_replica[name] = own
        validation[name] = dict(seed=seed, samples=len(v['labels']), correct=int(np.sum(v['predictions'] == v['labels'])),
                                accuracy=float(np.mean(v['predictions'] == v['labels'])))
    return dict(source=source, paired=paired, partition=partition, by_replica=by_replica, validation=validation)


def table_09(bundle: dict) -> tuple[list[dict], dict]:
    rows, detail = [], {}
    for name, seed in REPLICAS.items():
        own = bundle['by_replica'][name]
        part = next(p for p in bundle['partition'] if p['seed'] == seed and p['scope'] == 'complete_panel')
        require(part['sources'] == 100 and part['neighbors'] == 1297705, 'Readout partition scope differs')
        counts = {key: sum(r[key] for r in own.values()) for key in OUTCOMES}
        require(part['trace_changed_neighbors'] == counts['T1'] + counts['T2'] + counts['T3'], 'Trace-changed total differs from T1+T2+T3')
        require(part['final_trace_changed_counts_preserved'] + part['earlier_trace_changed_final_trace_preserved'] == counts['T1'],
                'T1 does not split into final-layer and earlier-only changes')
        correct = [r for r in own.values() if r['clean_correct']]
        robust = [r for r in correct if r['T3'] == 0]
        detail[name] = dict(seed=seed, **counts,
                            final_layer_trace_changed_counts_preserved=part['final_trace_changed_counts_preserved'],
                            earlier_only_trace_changes=part['earlier_trace_changed_final_trace_preserved'],
                            trace_changed_neighbors=part['trace_changed_neighbors'],
                            final_layer_fraction_of_trace_changes=part['final_trace_changed_counts_preserved'] / part['trace_changed_neighbors'],
                            earlier_only_fraction_of_trace_changes=part['earlier_trace_changed_final_trace_preserved'] / part['trace_changed_neighbors'],
                            initially_correct_sources=len(correct), robust_correct_sources=len(robust),
                            adverse_sensitive_sources=sum(r['adverse'] > 0 for r in correct),
                            sources_with_final_layer_changes=part['sources'])
    for key in ('T0', 'T1', 'T2', 'T3', 'final_layer_trace_changed_counts_preserved', 'adverse', 'initially_correct_sources', 'robust_correct_sources'):
        rows.append(dict(outcome=ROW_LABELS[key], replica_A=detail['A'][key], replica_B=detail['B'][key]))
    return rows, detail


def paired_bootstrap(bundle: dict, cohorts: dict, repetitions: int, seed: int) -> dict:
    """Paired within-label percentile source bootstrap of B-minus-A equal-source rate differences.

    One generator seeded once; cohorts are visited in the declared order, labels in ascending order,
    sources in ascending index order; each label stratum draws its ``repetitions x quota`` index
    block in one call, so the sampled index matrix is shared by the T3 and adverse differences.
    """
    rows = bundle['paired']; rng = np.random.default_rng(seed); result = {}
    for cohort in cohorts['cohort_order']:
        ids = set(cohorts['cohorts'][cohort]['source_ids'])
        members = sorted((r for r in rows if r['source_index'] in ids), key=lambda r: (r['label'], r['source_index']))
        require(len(members) == cohorts['cohorts'][cohort]['sources'], f'{cohort}: cohort size differs from the record')
        labels = np.array([r['label'] for r in members])
        arrays = {k: np.array([r[k] for r in members], dtype=np.float64)
                  for k in ('A_T3_source_rate', 'B_T3_source_rate', 'A_adverse_source_rate', 'B_adverse_source_rate')}
        blocks = []
        for label in sorted(set(labels.tolist())):
            group = np.flatnonzero(labels == label)
            blocks.append(group[rng.integers(0, len(group), size=(repetitions, len(group)))])
        index = np.concatenate(blocks, axis=1)
        entry = dict(sources=len(members), label_strata=len(set(labels.tolist())))
        for metric in ('T3', 'adverse'):
            observed = 100 * (arrays[f'B_{metric}_source_rate'].mean() - arrays[f'A_{metric}_source_rate'].mean())
            samples = 100 * (arrays[f'B_{metric}_source_rate'][index].mean(axis=1) - arrays[f'A_{metric}_source_rate'][index].mean(axis=1))
            low, high = np.percentile(samples, [2.5, 97.5])
            entry[f'{metric}_rate_difference_B_minus_A_percentage_points'] = float(observed)
            entry[f'{metric}_rate_difference_ci95'] = [float(low), float(high)]
        result[cohort] = entry
    return result


def analyze(root: Path) -> dict:
    bundle = load(root)
    contract = read_json(root / paths.REPLICA_CONTRACT)
    summary = read_json(root / paths.REPLICA_SUMMARY)
    cohorts = read_json(root / paths.REPLICA_COHORTS)
    records = read_json(root / paths.RECORDS)
    require(canonical_hash(contract) == summary['contract_sha256'], 'Replica summary does not point at the distributed contract')
    require(summary['original_contract_sha256'] == records['replica_contract_hashes']['original'], 'Original replica contract hash is not recorded')
    require(summary['status'] == 'COMPLETE' and summary['integrity']['complete_panel_pass'] is True, 'Replica audit summary is not complete')
    require([c['seed'] for c in contract['checkpoints']] == list(REPLICAS.values()), 'Replica seeds differ from the contract')
    for name, seed in REPLICAS.items():
        seed_result = summary['seed_results'][str(seed)]
        require(seed_result['clean_correct_sources'] == sum(r['clean_correct'] for r in bundle['by_replica'][name].values()), 'Summary clean-correct count differs')
    table, detail = table_09(bundle)
    for name, seed in REPLICAS.items():
        require(summary['seed_results'][str(seed)]['final_trace_changed_count_preserved'] == detail[name]['final_layer_trace_changed_counts_preserved'],
                'Summary final-layer count differs from the readout partition')
    paired = bundle['paired']
    four = [r for r in paired if not r['A_correct'] and r['B_correct'] and r['B_adverse'] > 0]
    four_ids = sorted(r['source_index'] for r in four)
    other = [r for r in paired if r['source_index'] not in four_ids]
    common = [r for r in paired if r['A_correct'] and r['B_correct']]
    require(sorted(cohorts['cohorts']['both_correct']['source_ids']) == sorted(r['source_index'] for r in common), 'Common-correct cohort differs')
    require(sorted(cohorts['cohorts']['A_wrong_B_correct']['source_ids']) == sorted(r['source_index'] for r in paired if not r['A_correct'] and r['B_correct']),
            'A-wrong/B-correct cohort differs')
    bootstrap = paired_bootstrap(bundle, cohorts, cohorts['bootstrap_repetitions'], cohorts['bootstrap_seed'])
    comparison = dict(
        validation=bundle['validation'],
        validation_accuracy_difference_B_minus_A_percentage_points=100 * (bundle['validation']['B']['accuracy'] - bundle['validation']['A']['accuracy']),
        class_change_ratio_B_over_A=detail['B']['T3'] / detail['A']['T3'],
        four_source_ids=four_ids,
        four_sources_T3=dict(A=sum(r['A_T3'] for r in four), B=sum(r['B_T3'] for r in four)),
        other_sources=len(other), other_sources_T3=dict(A=sum(r['A_T3'] for r in other), B=sum(r['B_T3'] for r in other)),
        four_sources_B_adverse=sum(r['B_adverse'] for r in four), B_adverse_total=detail['B']['adverse'],
        common_clean_correct_sources=len(common),
        common_clean_correct_adverse=dict(A=sum(r['A_adverse'] for r in common), B=sum(r['B_adverse'] for r in common)),
        common_clean_correct_A_source_equal_adverse_rate=float(np.mean([r['A_adverse_source_rate'] for r in common])),
        common_clean_correct_B_source_equal_adverse_rate=float(np.mean([r['B_adverse_source_rate'] for r in common])),
        common_clean_correct_difference_percentage_points=100 * float(np.mean([r['B_adverse_source_rate'] for r in common]) - np.mean([r['A_adverse_source_rate'] for r in common])),
        adverse_sensitive_of_initially_correct=dict(A=[detail['A']['adverse_sensitive_sources'], detail['A']['initially_correct_sources']],
                                                    B=[detail['B']['adverse_sensitive_sources'], detail['B']['initially_correct_sources']]),
        paired_bootstrap=bootstrap, bootstrap_repetitions=cohorts['bootstrap_repetitions'], bootstrap_seed=cohorts['bootstrap_seed'])
    return dict(replicas=detail, table_09=table, comparison=comparison, decisions=summary['decisions'],
                contracts=dict(replica_audit=summary['contract_sha256'], replica_audit_original=summary['original_contract_sha256']),
                new_model_inference=False)


def run(root: Path, out: Path, progress=print) -> dict:
    progress('[1/2] Checking paired replica cohorts and recomputing Table 9', flush=True)
    result = analyze(root)
    expected = read_json(root / paths.REFERENCE['replicas'])
    differences = compare_tree({k: result[k] for k in expected}, expected, atol=1e-12)
    write_json(out / 'replicas_numerical_agreement.json', dict(status='FAIL' if differences else 'PASS', differences=differences))
    progress('[2/2] Writing Table 9, the paired bootstrap and the four-source decomposition', flush=True)
    write_csv(out / 'table_09.csv', result['table_09'])
    c = result['comparison']
    write_csv(out / 'replicas_paired_bootstrap.csv', [dict(cohort=name, metric=metric, sources=entry['sources'],
                                                          difference_B_minus_A_percentage_points=entry[f'{metric}_rate_difference_B_minus_A_percentage_points'],
                                                          ci95_low=entry[f'{metric}_rate_difference_ci95'][0], ci95_high=entry[f'{metric}_rate_difference_ci95'][1],
                                                          repetitions=c['bootstrap_repetitions'], seed=c['bootstrap_seed'])
                                                     for name, entry in c['paired_bootstrap'].items() for metric in ('T3', 'adverse')])
    write_csv(out / 'replicas_four_source_decomposition.csv', [
        dict(cohort='four sources clean-correct only in B', sources=4, A_T3=c['four_sources_T3']['A'], B_T3=c['four_sources_T3']['B'],
             A_adverse='', B_adverse=c['four_sources_B_adverse']),
        dict(cohort='other sources', sources=c['other_sources'], A_T3=c['other_sources_T3']['A'], B_T3=c['other_sources_T3']['B'],
             A_adverse='', B_adverse=c['B_adverse_total'] - c['four_sources_B_adverse']),
        dict(cohort='clean-correct in both', sources=c['common_clean_correct_sources'], A_T3='', B_T3='',
             A_adverse=c['common_clean_correct_adverse']['A'], B_adverse=c['common_clean_correct_adverse']['B'])])
    require(not differences, 'Replica reference comparison failed: ' + '; '.join(differences[:10]))
    report = dict(status='PASS', **result)
    write_json(out / 'replicas.json', report)
    return report
