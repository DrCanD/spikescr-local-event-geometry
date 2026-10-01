"""Search policies on the retained map (manuscript §4.3, Table 4, Supplementary S2).

Randomized policies are exact finite-population expectations under uniform-within-source sampling
at fixed total candidate-query budgets; gradient rankings are deterministic and are looked up in the
complete map after ranking. No model is executed.
"""
from __future__ import annotations
import hashlib
import json
from fractions import Fraction
from pathlib import Path
import numpy as np
from scipy.stats import hypergeom
from . import paths
from .io import compare_tree, load_npz, read_json, read_numeric_csv, write_csv, write_json


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def contract_hash(root: Path, relative: str) -> str:
    """Hash of a recorded contract payload (sorted compact JSON), checked against the stored value."""
    record = read_json(root / relative)
    digest = hashlib.sha256(json.dumps(record['payload'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    require(digest == record['contract_sha256'], f'{relative}: contract hash mismatch')
    return digest


# ---------------------------------------------------------------- exact finite-population coverage
def allocate_exact(total, capacities, weights, source_ids):
    """Capped proportional allocation with exact rational largest-remainder rounding (source-id tie break)."""
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
    """Probability that uniform sampling of ``queries`` candidates hits at least one adverse candidate."""
    p = hypergeom.sf(0, capacities, adverse, queries)
    for i in np.flatnonzero(adverse > 0):  # independent product check
        n, a, q = int(capacities[i]), int(adverse[i]), int(queries[i])
        direct = 1.0 if q > n - a else float(-np.expm1(np.log1p(-float(q) / (n - np.arange(a))).sum()))
        require(bool(np.isclose(p[i], direct, rtol=2e-10, atol=2e-12)), 'Coverage counting implementations differ')
    return p


# ---------------------------------------------------------------- the analysis
def analyze(root: Path) -> dict:
    rows = sorted((r for r in read_numeric_csv(root / paths.SOURCES) if r['clean_correct']), key=lambda r: r['source_index'])
    ids = np.array([r['source_index'] for r in rows])
    n = np.array([r['unique_candidates'] for r in rows])
    a = np.array([r['adverse_unique'] for r in rows])
    margin = np.array([r['clean_margin'] for r in rows])
    horizon = np.array([r['horizon_steps'] for r in rows])
    require((len(rows), int(n.sum()), int(a.sum()), int((a > 0).sum())) == (84, 628171, 5157, 13), 'Search comparison population differs')
    spec = read_json(root / paths.SEARCH_POLICIES)['cutoff_sensitivity']
    coverage = []
    for ranking, values in [('raw_margin', margin), ('margin_per_horizon', margin / horizon)]:
        order = np.lexsort((ids, values))
        ranks = np.empty(len(ids), dtype=int)
        ranks[order] = np.arange(1, len(ids) + 1)
        policies = [('uniform', np.ones(len(ids), dtype=int))]
        for percent in spec['percent_thresholds']:
            selected_count = int(np.ceil(len(ids) * percent / 100))
            policies.append((f'lowest_{percent}_percent', (ranks <= selected_count).astype(int)))
        policies.append(('half_uniform_half_quartile', np.ones(len(ids), dtype=int) + 4 * (ranks <= 21)))
        for policy, weights in policies:
            for budget in spec['total_candidate_query_budgets']:
                q, overflow = allocate_exact(budget, n, weights, ids)
                p = detection(n, a, q)
                coverage.append(dict(ranking=ranking, policy=policy, total_candidate_queries=budget,
                                     expected_detected_sources=float(p.sum()), probability_detect_all13=float(p[a > 0].prod()),
                                     uniform_overflow=overflow, source1627_rank=int(ranks[ids == 1627][0]),
                                     source1627_queries=int(q[ids == 1627][0]),
                                     vulnerable_sources_in_lowest_quartile=int(((ranks <= 21) & (a > 0)).sum()),
                                     vulnerable_sources_in_lowest_half=int(((ranks <= 42) & (a > 0)).sum())))
    # the recorded cutoff-sensitivity table (Supplementary S2) must agree with the recomputation
    recorded = read_numeric_csv(root / paths.SEARCH_CUTOFF_SENSITIVITY)
    for r in recorded:
        match = [c for c in coverage if c['ranking'] == r['ranking'] and c['policy'] == f"lowest_{r['threshold_percent']}_percent"
                 and c['total_candidate_queries'] == r['total_candidate_queries']]
        require(len(match) == 1 and abs(match[0]['expected_detected_sources'] - r['expected_detected_sources']) < 1e-9
                and abs(match[0]['probability_detect_all13'] - r['probability_detect_all13']) < 1e-12
                and match[0]['source1627_rank'] == r['source1627_rank'] and match[0]['source1627_queries'] == r['source1627_queries'],
                'Recorded cutoff-sensitivity row differs from the recomputation')
    saved = load_npz(root / paths.SEARCH_RANKS)
    policy_scores = load_npz(root / paths.SEARCH_SCORES)
    gradients = load_npz(root / paths.SEARCH_RUNNER_UP_GRADIENTS)
    for extra in (policy_scores, gradients):
        require(np.array_equal(extra['source_index'], saved['source_index']), 'Ranking/gradient source alignment differs')
    require(np.array_equal(policy_scores['source_offsets'], saved['source_offsets']) and
            np.array_equal(policy_scores['policy_names'], saved['policy_names']), 'Policy score alignment differs')
    saved.update(move_scores=policy_scores['move_scores'], runner_up_input_gradients=gradients['runner_up_input_gradients'],
                 gradient_offsets=gradients['gradient_offsets'], horizon=gradients['horizon'])
    geometry = load_npz(root / paths.CANDIDATES)
    panel = load_npz(root / paths.PANEL_INPUTS)
    require(np.array_equal(saved['source_index'], np.unique(geometry['source_index'])), 'Ranking source cohort differs')
    require(saved['source_offsets'].shape == (101,) and saved['source_offsets'][0] == 0 and saved['source_offsets'][-1] == 725070
            and bool(np.all(np.diff(saved['source_offsets']) > 0)), 'Invalid source offsets')
    require(saved['move_scores'].shape == saved['ranked_candidate_index'].shape == (3, 725070), 'Ranking arrays have invalid shapes')
    require(bool(np.isfinite(saved['move_scores']).all()), 'Nonfinite ranking scores')
    policy_names = list(map(str, saved['policy_names']))
    require(policy_names == ['runner_up', 'max_directional_gain', 'max_predicted_target_margin'], 'Ranking policies differ')
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
        directional = grad[geometry['to_bin'][selected], geometry['feature'][selected]] - grad[geometry['from_bin'][selected], geometry['feature'][selected]]
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
            discovery.append(dict(policy=policy, budget_per_source=budget, total_candidate_queries=sum(r['candidate_queries'] for r in b),
                                  actual_discovered_sources=sum(r['discovered'] for r in b),
                                  selected_adverse_candidates=sum(r['selected_adverse'] for r in b)))
        f = [r for r in first_rows if r['policy'] == policy]
        prefix = max(r['first_adverse_rank'] for r in f if r['first_adverse_rank'] is not None)
        queries = sum(min(r['capacity'], prefix) for r in f)
        prefixes.append(dict(policy=policy, posthoc_prefix_per_source=prefix, capped_queries=queries,
                             fraction_of_exhaustive=queries / int(n.sum()),
                             sources_with_shorter_neighborhood=sum(r['capacity'] < prefix for r in f)))
    # the recorded all-target summary (from the gradient run) must agree with the lookup
    summary = read_json(root / paths.SEARCH_SUMMARY)
    for row in summary['budgets']:
        match = [d for d in discovery if d['policy'] == row['policy'] and d['budget_per_source'] == row['budget_per_source']]
        require(len(match) == 1 and match[0]['actual_discovered_sources'] == row['actual_discovered_sources']
                and match[0]['selected_adverse_candidates'] == row['selected_adverse_candidates'], 'Recorded all-target discovery differs')
    witnesses = read_json(root / paths.SEARCH_WITNESSES)
    witness_check = {}
    ru = witnesses['runner_up']
    require(ru['all_prediction_match'] is True and ru['all_adverse'] is True and len(ru['rows']) == 13, 'Runner-up witness record incomplete')
    for w in ru['rows']:
        first = [r for r in first_rows if r['policy'] == 'runner_up' and r['source_index'] == w['source_index']]
        require(len(first) == 1 and first[0]['first_adverse_rank'] == w['first_adverse_rank'] and w['prediction_match'] and w['is_adverse'],
                'Runner-up witness rank differs from the frozen ranking')
    witness_check['runner_up'] = dict(rows=len(ru['rows']), all_predictions_match=True)
    at = witnesses['all_target']
    require(at['all_predictions_match'] is True and len(at['rows']) == 26, 'All-target witness record incomplete')
    for w in at['rows']:
        first = [r for r in first_rows if r['policy'] == w['policy'] and r['source_index'] == w['source_index']]
        require(len(first) == 1 and first[0]['first_adverse_rank'] == w['rank'] and w['first_adverse_prediction_reproduced'] is True,
                'All-target witness rank differs from the frozen ranking')
    witness_check['all_target'] = dict(rows=len(at['rows']), all_predictions_match=True)
    contracts = dict(gradient_runner_up=contract_hash(root, paths.CONTRACT_GRADIENT_RUNNER_UP),
                     gradient_all_target=contract_hash(root, paths.CONTRACT_GRADIENT_ALL_TARGET))
    require(contracts['gradient_all_target'] == summary['contract_sha256'], 'All-target summary contract differs')
    return dict(initially_correct_sources=84, exhaustive_candidate_queries=int(n.sum()),
                finite_sampling_coverage=coverage, deterministic_discovery=discovery, first_adverse_ranks=first_rows,
                posthoc_full_discovery_prefixes=prefixes, witness_verification=witness_check, contracts=contracts,
                retrospective_policy_comparison=True, new_model_inference=False)


# ---------------------------------------------------------------- Table 4 and the cutoff sweep
POLICY_LABELS = {'uniform': 'Uniform', 'lowest_10_percent': 'Lowest 10%', 'lowest_25_percent': 'Lowest 25%',
                 'lowest_50_percent': 'Lowest 50%', 'half_uniform_half_quartile': '50/50',
                 'runner_up': 'Runner-up', 'max_directional_gain': 'All-target gain', 'max_predicted_target_margin': 'All-target margin'}


def table_04(result: dict) -> list[dict]:
    rows = []
    for budget in (840, 8400, 84000):
        row = dict(candidate_queries=budget)
        for policy in ('uniform', 'lowest_10_percent', 'lowest_25_percent', 'lowest_50_percent', 'half_uniform_half_quartile'):
            c = next(c for c in result['finite_sampling_coverage'] if c['ranking'] == 'raw_margin' and c['policy'] == policy and c['total_candidate_queries'] == budget)
            row[f'{policy}_expected'] = c['expected_detected_sources']
        for policy in ('runner_up', 'max_directional_gain', 'max_predicted_target_margin'):
            d = next(d for d in result['deterministic_discovery'] if d['policy'] == policy and d['total_candidate_queries'] == budget)
            row[f'{policy}_discovered'] = d['actual_discovered_sources']
        rows.append(row)
    return rows


def run(root: Path, out: Path, progress=print) -> dict:
    progress('[1/2] Recomputing finite-sampling expectations and frozen gradient rankings', flush=True)
    result = analyze(root)
    expected = read_json(root / paths.REFERENCE['search'])
    comparable = {k: result[k] for k in expected}
    differences = compare_tree(comparable, expected, atol=1e-12)
    write_json(out / 'search_numerical_agreement.json', dict(status='FAIL' if differences else 'PASS', differences=differences))
    progress('[2/2] Writing Table 4, prefix costs, first adverse ranks and the cutoff sweep', flush=True)
    write_csv(out / 'table_04.csv', table_04(result))
    write_csv(out / 'search_prefix_costs.csv', result['posthoc_full_discovery_prefixes'])
    write_csv(out / 'search_first_adverse_ranks.csv', result['first_adverse_ranks'])
    write_csv(out / 'search_cutoff_sensitivity.csv', [c for c in result['finite_sampling_coverage']])
    write_csv(out / 'search_deterministic_discovery.csv', result['deterministic_discovery'])
    require(not differences, 'Search reference comparison failed: ' + '; '.join(differences[:10]))
    report = dict(status='PASS', **result)
    write_json(out / 'search.json', report)
    return report
