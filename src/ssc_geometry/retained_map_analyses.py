"""Finite search coverage and paired count-readout replicas from retained records."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom

from .execution_controls import require
from .io import compare_tree, load_npz, read_json, read_numeric_csv, write_csv, write_json


def allocate_exact(total, capacities, weights, source_ids):
    """Recorded capped proportional policy with exact rational remainder ties."""
    capacities = np.asarray(capacities, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.int64)
    require(0 <= total <= int(capacities.sum()), 'Allocation budget outside capacity')
    require(bool(np.all(weights >= 0)) and int(weights.sum()) > 0, 'Invalid policy weights')
    real = [Fraction(0) for _ in capacities]
    active = capacities > 0
    remaining = int(total)
    uniform_overflow = False
    while remaining > 0:
        w = weights * active
        if w.sum() == 0:
            w = active.astype(np.int64)
            uniform_overflow = True
        denominator = int(w.sum())
        proposal = [Fraction(remaining * int(w[i]), denominator) for i in range(len(w))]
        saturated = active & np.array([proposal[i] >= int(capacities[i]) for i in range(len(w))])
        if saturated.any():
            for i in np.flatnonzero(saturated):
                real[i] = Fraction(int(capacities[i]))
            remaining -= int(capacities[saturated].sum())
            active[saturated] = False
        else:
            for i in np.flatnonzero(active):
                real[i] = proposal[i]
            remaining = 0
    counts = np.array([v.numerator // v.denominator for v in real], dtype=np.int64)
    left = int(total - counts.sum())
    eligible = np.flatnonzero(counts < capacities)
    order = sorted(eligible, key=lambda i: (-(real[i] - int(counts[i])), int(source_ids[i])))
    require(left <= len(order), 'Invalid allocation remainder')
    counts[order[:left]] += 1
    require(int(counts.sum()) == total and bool(np.all(counts <= capacities)), 'Allocation conservation failed')
    return counts, uniform_overflow


def detection(capacities, adverse, queries):
    p = hypergeom.sf(0, capacities, adverse, queries)
    # Independent product check, using adverse counts instead of a CDF.
    for i in np.flatnonzero(adverse > 0):
        n, a, q = int(capacities[i]), int(adverse[i]), int(queries[i])
        direct = 1.0 if q > n - a else float(-np.expm1(np.log1p(-float(q) / (n - np.arange(a))).sum()))
        require(bool(np.isclose(p[i], direct, rtol=2e-10, atol=2e-12)), 'Coverage counting implementations differ')
    return p


def search_analysis(root: Path) -> dict:
    rows = sorted((r for r in read_numeric_csv(root / 'data/neighborhood/source_geometry.csv')
                   if r['clean_correct']), key=lambda r: r['source_index'])
    ids = np.array([r['source_index'] for r in rows])
    n = np.array([r['unique_candidates'] for r in rows])
    a = np.array([r['adverse_unique'] for r in rows])
    margin = np.array([r['clean_margin'] for r in rows])
    horizon = np.array([r['horizon_steps'] for r in rows])
    require((len(rows), int(n.sum()), int(a.sum()), int((a > 0).sum())) == (84, 628171, 5157, 13),
            'Search comparison population differs')
    threshold = read_json(root / 'data/search_policies/threshold_spec.json')
    coverage = []
    for ranking, values in [('raw_margin', margin), ('margin_per_horizon', margin / horizon)]:
        order = np.lexsort((ids, values))
        ranks = np.empty(len(ids), dtype=int)
        ranks[order] = np.arange(1, len(ids) + 1)
        policies = [('uniform', np.ones(len(ids), dtype=int))]
        for percent in threshold['percent_thresholds']:
            selected_count = int(np.ceil(len(ids) * percent / 100))
            policies.append((f'lowest_{percent}_percent', (ranks <= selected_count).astype(int)))
        policies.append(('half_uniform_half_quartile', np.ones(len(ids), dtype=int) + 4 * (ranks <= 21)))
        for policy, weights in policies:
            for budget in threshold['total_candidate_query_budgets']:
                q, overflow = allocate_exact(budget, n, weights, ids)
                p = detection(n, a, q)
                coverage.append(dict(ranking=ranking, policy=policy, total_candidate_queries=budget,
                                     expected_detected_sources=float(p.sum()),
                                     probability_detect_all13=float(p[a > 0].prod()),
                                     uniform_overflow=overflow, source1627_rank=int(ranks[ids == 1627][0]),
                                     source1627_queries=int(q[ids == 1627][0])))
    saved = load_npz(root / 'data/search_policies/candidate_rankings.npz')
    policy_scores = load_npz(root / 'data/search_policies/candidate_policy_scores.npz')
    gradients = load_npz(root / 'data/search_policies/runner_up_input_gradients.npz')
    for extra in (policy_scores, gradients):
        require(np.array_equal(extra['source_index'], saved['source_index']), 'Ranking/gradient source alignment differs')
    require(np.array_equal(policy_scores['source_offsets'], saved['source_offsets']) and
            np.array_equal(policy_scores['policy_names'], saved['policy_names']), 'Policy score alignment differs')
    saved.update(move_scores=policy_scores['move_scores'],
                 runner_up_input_gradients=gradients['runner_up_input_gradients'],
                 gradient_offsets=gradients['gradient_offsets'], horizon=gradients['horizon'])
    geometry = load_npz(root / 'data/neighborhood/candidate_records.npz')
    panel = load_npz(root / 'data/panel/inputs.npz')
    require(np.array_equal(saved['source_index'], np.unique(geometry['source_index'])), 'Ranking source cohort differs')
    require(saved['source_offsets'].shape == (101,) and saved['source_offsets'][0] == 0 and
            saved['source_offsets'][-1] == 725070 and bool(np.all(np.diff(saved['source_offsets']) > 0)),
            'Invalid source offsets')
    require(saved['move_scores'].shape == saved['ranked_candidate_index'].shape == (3, 725070),
            'Ranking arrays have invalid shapes')
    require(bool(np.isfinite(saved['move_scores']).all()), 'Nonfinite ranking scores')
    policy_names = list(map(str, saved['policy_names']))
    require(policy_names == ['runner_up', 'max_directional_gain', 'max_predicted_target_margin'],
            'Ranking policies differ')
    budget_rows, first_rows = [], []
    for i, sid in enumerate(saved['source_index']):
        selected = geometry['source_index'] == sid
        candidate_ids = geometry['candidate_index'][selected]
        start, end = saved['source_offsets'][i:i + 2]
        require(np.array_equal(candidate_ids, np.arange(end - start)), 'Candidate ranking identity differs')
        gs, ge = saved['gradient_offsets'][i:i + 2]
        H = int(saved['horizon'][i])
        grad = saved['runner_up_input_gradients'][gs:ge].reshape(H, 140)
        require(panel[f'source_{sid:05d}'].shape == grad.shape, 'Gradient horizon differs from panel input')
        directional = grad[geometry['to_bin'][selected], geometry['feature'][selected]] - \
                      grad[geometry['from_bin'][selected], geometry['feature'][selected]]
        require(np.array_equal(directional, saved['move_scores'][0, start:end]), 'Runner-up directional scores differ')
        if sid not in ids:
            continue
        outcome = geometry['transition_code'][selected]
        for j, policy in enumerate(policy_names):
            rank = saved['ranked_candidate_index'][j, start:end]
            expected = np.lexsort((candidate_ids, -saved['move_scores'][j, start:end]))
            require(np.array_equal(rank, expected), 'Saved candidate rank or tie break differs')
            hits = np.flatnonzero(outcome[rank] == 1)
            first_rows.append(dict(policy=policy, source_index=int(sid), capacity=len(rank),
                                   first_adverse_rank=int(hits[0] + 1) if len(hits) else None))
            for budget in (1, 10, 100, 1000):
                chosen = outcome[rank[:budget]] == 1
                budget_rows.append(dict(policy=policy, budget_per_source=budget, candidate_queries=len(chosen),
                                        discovered=bool(chosen.any()), selected_adverse=int(chosen.sum())))
    discovery, prefixes = [], []
    for policy in policy_names:
        for budget in (1, 10, 100, 1000):
            b = [r for r in budget_rows if r['policy'] == policy and r['budget_per_source'] == budget]
            require(len(b) == 84, 'Incomplete ranking coverage')
            discovery.append(dict(policy=policy, budget_per_source=budget,
                                  total_candidate_queries=sum(r['candidate_queries'] for r in b),
                                  actual_discovered_sources=sum(r['discovered'] for r in b),
                                  selected_adverse_candidates=sum(r['selected_adverse'] for r in b)))
        f = [r for r in first_rows if r['policy'] == policy]
        prefix = max(r['first_adverse_rank'] for r in f if r['first_adverse_rank'] is not None)
        queries = sum(min(r['capacity'], prefix) for r in f)
        prefixes.append(dict(policy=policy, posthoc_prefix_per_source=prefix, capped_queries=queries,
                             fraction_of_exhaustive=queries / int(n.sum()),
                             sources_with_shorter_neighborhood=sum(r['capacity'] < prefix for r in f)))
    return dict(initially_correct_sources=84, exhaustive_candidate_queries=int(n.sum()),
                finite_sampling_coverage=coverage, deterministic_discovery=discovery,
                first_adverse_ranks=first_rows, posthoc_full_discovery_prefixes=prefixes,
                retrospective_policy_comparison=True, new_model_inference=False)


def replica_analysis(root: Path) -> dict:
    folder = root / 'data/replicas'
    source = read_numeric_csv(folder / 'source_results.csv')
    paired = read_numeric_csv(folder / 'paired_source_data.csv')
    require(len(source) == 200 and len(paired) == 100, 'Replica panel cardinality differs')
    summary, reference = {}, {}
    for name, seed in [('A', 341969035), ('B', 554720563)]:
        own = {r['source_index']: r for r in source if r['seed'] == seed}
        require(len(own) == 100, 'Replica source identities differ')
        for r in paired:
            s = own[r['source_index']]
            require(s['complete'] and s['candidates_done'] == r['neighbors'] == s['candidates'],
                    'Replica source census is incomplete')
            require(bool(s['clean_correct']) == r[f'{name}_correct'], 'Replica clean cohort differs')
            for key in ('T0', 'T1', 'T2', 'T3', 'adverse', 'corrective', 'lateral'):
                require(s[key] == r[f'{name}_{key}'], 'Replica paired counts differ')
        validation = load_npz(folder / f'validation_replica_{name}.npz')
        require(np.array_equal(validation['source_indices'], np.arange(9981)), 'Replica validation source order differs')
        require(np.array_equal(validation['logits'].argmax(axis=1), validation['predictions']),
                'Replica saved validation prediction disagrees with scores')
        reference[name] = own
        summary[name] = dict(seed=seed, validation_correct=int(np.sum(validation['predictions'] == validation['labels'])),
                             validation_samples=len(validation['labels']),
                             validation_accuracy=float(np.mean(validation['predictions'] == validation['labels'])),
                             T3=sum(r['T3'] for r in own.values()), adverse=sum(r['adverse'] for r in own.values()))
    four = [r for r in paired if not r['A_correct'] and r['B_correct'] and r['B_adverse'] > 0]
    common = [r for r in paired if r['A_correct'] and r['B_correct']]
    A = float(np.mean([r['A_adverse_source_rate'] for r in common]))
    B = float(np.mean([r['B_adverse_source_rate'] for r in common]))
    return dict(replicas=summary, class_change_ratio_B_over_A=summary['B']['T3'] / summary['A']['T3'],
                four_source_ids=[r['source_index'] for r in four],
                four_sources_B_adverse=sum(r['B_adverse'] for r in four),
                common_clean_correct_sources=len(common),
                common_clean_correct_A_source_equal_adverse_rate=A,
                common_clean_correct_B_source_equal_adverse_rate=B,
                common_clean_correct_difference_percentage_points=100 * (B - A),
                new_model_inference=False)


def reproduce(root: Path, out: Path) -> dict:
    print('[1/3] Recomputing finite sampling and frozen gradient search policies', flush=True)
    search = search_analysis(root)
    print('[2/3] Checking paired replica cohorts and four-source concentration', flush=True)
    replicas = replica_analysis(root)
    report = dict(search_policies=search, count_readout_replicas=replicas)
    expected = read_json(root / 'data/reference/extended_analysis_statistics.json')
    differences = compare_tree(report, expected, atol=1e-12)
    write_json(out / 'extended_analysis_statistics.json', report)
    write_json(out / 'numerical_agreement.json', dict(status='FAIL' if differences else 'PASS', differences=differences))
    for name in ('finite_sampling_coverage', 'deterministic_discovery', 'first_adverse_ranks',
                 'posthoc_full_discovery_prefixes'):
        write_csv(out / f'{name}.csv', search[name])
    require(not differences, 'Extended-analysis reference comparison failed: ' + '; '.join(differences[:10]))
    print('[3/3] Finished retained-map analyses; no model inference executed', flush=True)
    return dict(status='PASS', initially_correct_sources=84, replicas=2, new_model_inference=False)
