"""Recompute the gradient search rankings of §4.3 on the frozen checkpoint (model inference; CPU or GPU).

Steps (``--step``), resumable per source:

  gate        100 panel sources: clean singleton scores must match data/panel/clean_scores.npz (score tolerance 1e-4,
              same predicted class); every LIF spike tensor is hashed so the surrogate forward can be checked bit for bit
  runner-up   one backward per source: objective = score(clean runner-up) - score(clean predicted class);
              move score = gradient[to, feature] - gradient[from, feature]; ranking descending with candidate-index tie break
  all-target  34 backwards per source (every class other than the clean prediction); policies
              max_directional_gain = max over targets of the directional gain, and
              max_predicted_target_margin = max over targets of (clean target margin + directional gain)
  compare     move scores and rankings of this run against data/search/

The LIF nodes run in training mode for the backward (public ATan surrogate); batch normalisation and dropout
stay in evaluation mode, and the surrogate forward must reproduce the evaluation forward bit for bit before a
gradient is accepted. Candidate outcomes are never read while ranking; they are looked up afterwards by
``python -m ssc_geometry search``.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ssc_geometry import paths, __version__
from ssc_geometry.integrity import verify_integrity
from ssc_geometry.io import assert_output_safe, load_npz, read_json, write_json, sha256_file

STEPS = ('gate', 'runner-up', 'all-target', 'compare')
SCORE_ATOL = 1e-4
POLICIES = ('runner_up', 'max_directional_gain', 'max_predicted_target_margin')


def json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_npz(path: Path, **arrays) -> None:
    temporary = path.with_name(path.name + '.tmp.npz'); np.savez_compressed(temporary, **arrays); temporary.replace(path)


def spike_hash_hooks(model):
    hashes, handles = {}, []

    def make_hook(name):
        def hook(_module, _inputs, output):
            hashes[name] = hashlib.sha256(output.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        return hook
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'LIFNode':
            handles.append(module.register_forward_hook(make_hook(name)))
    return hashes, handles


def clean_forward(model, forward, torch, x, device, capture=False):
    forward.reset_spiking_state(model)
    hashes, handles = spike_hash_hooks(model) if capture else ({}, [])
    try:
        value = torch.from_numpy(x[None].astype(np.float32)).to(device)
        with torch.no_grad():
            scores = forward.upstream_softmax_sum(model(value, torch.ones((1, len(x)), dtype=torch.bool, device=device)))[0]
        return scores.cpu().numpy().astype(np.float32), hashes
    finally:
        for handle in handles:
            handle.remove()
        forward.reset_spiking_state(model)


def surrogate_forward(model, forward, torch, x, device):
    """Forward with LIF nodes in training mode and gradient enabled; returns scores tensor, input tensor, hashes, cleanup."""
    model.eval(); forward.reset_spiking_state(model)
    lif_nodes = [m for m in model.modules() if m.__class__.__name__ == 'LIFNode']
    if not lif_nodes:
        raise ValueError('No LIF nodes found')
    for node in lif_nodes:
        node.train(True)
    if any(m.training for m in model.modules() if isinstance(m, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.Dropout))):
        raise ValueError('Batch normalisation or dropout was enabled')
    value = torch.from_numpy(x[None].astype(np.float32)).to(device).requires_grad_(True)
    hashes, handles = spike_hash_hooks(model)
    scores = forward.upstream_softmax_sum(model(value, torch.ones((1, len(x)), dtype=torch.bool, device=device)))[0]

    def cleanup():
        for handle in handles:
            handle.remove()
        model.eval(); forward.reset_spiking_state(model)
    return scores, value, hashes, cleanup


def source_candidates(geometry, sid, x, clean_prediction, label):
    mask = geometry['source_index'] == sid
    a = {k: v[mask] for k, v in geometry.items()}
    order = np.argsort(a['candidate_index'], kind='stable'); a = {k: v[order] for k, v in a.items()}; n = len(order)
    frm, to, feat = (a[k].astype(np.int64) for k in ('from_bin', 'to_bin', 'feature'))
    expected_n = 2 * np.count_nonzero(x) - np.count_nonzero(x[0]) - np.count_nonzero(x[-1])
    if n != expected_n or not np.array_equal(a['candidate_index'], np.arange(n)):
        raise ValueError('Candidate coverage differs from the panel input')
    if np.any(np.abs(to - frm) != 1) or np.any(frm < 0) or np.any(to < 0) or np.any(frm >= len(x)) or np.any(to >= len(x)) or np.any(feat < 0) or np.any(feat >= 140):
        raise ValueError('Illegal candidate coordinates')
    if np.any(x[frm, feat] <= 0) or not np.array_equal(x[frm, feat], a['multiplicity']):
        raise ValueError('Candidate multiplicities differ from the panel input')
    if np.any(a['label'] != label) or np.any(a['clean_prediction'] != clean_prediction):
        raise ValueError('Candidate labels or clean predictions differ')
    return a, frm, to, feat


# ---------------------------------------------------------------- steps
def gate(model, forward, torch, data, panel, device, out, contract, log):
    path = out / 'gate.npz'
    if path.exists():
        z = load_npz(path)
        if str(z['contract_sha256']) == contract and bool(z['pass']) and len(z['source_index']) == len(panel['source_index']):
            log('gate: reused'); return z
    ids = panel['source_index']; scores, errors, parity, rows, names = [], [], [], [], None
    for j, sid in enumerate(ids):
        x, label = data.get(int(sid))
        s, hashes = clean_forward(model, forward, torch, x, device, capture=True)
        if names is None:
            names = sorted(hashes)
        if sorted(hashes) != names:
            raise ValueError('LIF topology changed between sources')
        rows.append([hashes[n] for n in names]); scores.append(s)
        errors.append(float(np.max(np.abs(s - panel['scores'][j])))); parity.append(int(s.argmax()) == int(panel['scores'][j].argmax()))
        if (j + 1) % 10 == 0 or j + 1 == len(ids):
            log(f'gate {j + 1}/{len(ids)} max_abs={max(errors):.3g} class_mismatches={len(parity) - sum(parity)}')
    passed = all(parity) and max(errors) <= SCORE_ATOL
    atomic_npz(path, contract_sha256=np.asarray(contract), source_index=ids, scores=np.stack(scores), max_abs_error=np.asarray(errors),
               class_parity=np.asarray(parity), lif_names=np.asarray(names), lif_spike_sha256=np.asarray(rows), **{'pass': np.asarray(passed)})
    write_json(out / 'gate.json', dict(contract_sha256=contract, sources=len(ids), score_atol=SCORE_ATOL, max_abs_error=max(errors),
                                       class_disagreements=len(ids) - sum(parity), bit_exact_sources=sum(e == 0 for e in errors), **{'pass': bool(passed)}))
    if not passed:
        raise RuntimeError('Canonical 100-source parity failed; no gradient is computed')
    return load_npz(path)


def runner_up(model, forward, torch, data, panel, geometry, device, out, contract, log, gate_arrays):
    folder = out / 'runner_up'; folder.mkdir(exist_ok=True)
    runtime = {int(s): gate_arrays['scores'][j] for j, s in enumerate(gate_arrays['source_index'])}
    names = list(map(str, gate_arrays['lif_names'])); spikes = {int(s): list(map(str, gate_arrays['lif_spike_sha256'][j])) for j, s in enumerate(gate_arrays['source_index'])}
    for j, sid in enumerate(panel['source_index']):
        sid = int(sid); path = folder / f'source_{sid:05d}.npz'
        if path.exists():
            continue
        x, label = data.get(sid); ref = panel['scores'][j]; clean = int(ref.argmax())
        a, frm, to, feat = source_candidates(geometry, sid, x, clean, label)
        scores, value, hashes, cleanup = surrogate_forward(model, forward, torch, x, device)
        try:
            s = scores.detach().cpu().numpy().astype(np.float32)
            if not np.array_equal(s, runtime[sid]) or sorted(hashes) != names or [hashes[n] for n in names] != spikes[sid]:
                raise RuntimeError(f'Surrogate forward is not bit-identical to the evaluation forward for source {sid}')
            if int(s.argmax()) != clean or np.max(np.abs(s - ref)) > SCORE_ATOL:
                raise RuntimeError('Canonical score gate failed')
            others = ref.copy(); others[clean] = -np.inf; rival = int(others.argmax())
            grad = torch.autograd.grad(scores[rival] - scores[clean], value)[0][0].detach().cpu().numpy().astype(np.float32)
        finally:
            cleanup()
        if not np.isfinite(grad).all():
            raise ValueError('Nonfinite input gradient')
        move_score = grad[to, feat] - grad[frm, feat]
        ranking = np.lexsort((a['candidate_index'], -move_score))
        atomic_npz(path, contract_sha256=np.asarray(contract), source_index=np.asarray(sid), input_gradient=grad, move_score=move_score,
                   ranked_candidate_index=a['candidate_index'][ranking], surrogate_clean_scores=s, rival_class=np.asarray(rival), clean_class=np.asarray(clean))
        log(f'runner-up {j + 1}/{len(panel["source_index"])} source={sid} nonzero_gradient={np.count_nonzero(grad)}')


def all_target(model, forward, torch, data, panel, geometry, device, out, contract, log, gate_arrays):
    folder = out / 'all_target'; folder.mkdir(exist_ok=True)
    runtime = {int(s): gate_arrays['scores'][j] for j, s in enumerate(gate_arrays['source_index'])}
    names = list(map(str, gate_arrays['lif_names'])); spikes = {int(s): list(map(str, gate_arrays['lif_spike_sha256'][j])) for j, s in enumerate(gate_arrays['source_index'])}
    for j, sid in enumerate(panel['source_index']):
        sid = int(sid); path = folder / f'source_{sid:05d}.npz'
        if path.exists():
            continue
        started = time.monotonic(); x, label = data.get(sid); ref = panel['scores'][j]; clean = int(ref.argmax())
        a, frm, to, feat = source_candidates(geometry, sid, x, clean, label)
        scores, value, hashes, cleanup = surrogate_forward(model, forward, torch, x, device)
        try:
            s = scores.detach().cpu().numpy().astype(np.float32)
            if not np.array_equal(s, runtime[sid]) or sorted(hashes) != names or [hashes[n] for n in names] != spikes[sid]:
                raise RuntimeError(f'Surrogate forward is not bit-identical to the evaluation forward for source {sid}')
            if int(s.argmax()) != clean or np.max(np.abs(s - ref)) > SCORE_ATOL:
                raise RuntimeError('Canonical score gate failed')
            targets = np.array([c for c in range(35) if c != clean], dtype=np.int16); gains, gradients = [], []
            for k, target in enumerate(targets):
                grad = torch.autograd.grad(scores[int(target)] - scores[clean], value, retain_graph=k < 33)[0][0].detach().cpu().numpy().astype(np.float32)
                if not np.isfinite(grad).all():
                    raise ValueError('Nonfinite gradient')
                gradients.append(grad); gains.append(grad[to, feat] - grad[frm, feat])
        finally:
            cleanup()
        gains = np.stack(gains); margins = s[targets] - s[clean]
        directional = gains.max(axis=0); predicted = (gains + margins[:, None]).max(axis=0)
        ranks = np.stack([np.lexsort((a['candidate_index'], -v)) for v in (directional, predicted)])
        atomic_npz(path, contract_sha256=np.asarray(contract), source_index=np.asarray(sid), targets=targets, clean_class=np.asarray(clean), clean_scores=s,
                   input_gradients=np.stack(gradients), candidate_index=a['candidate_index'], target_directional_gains=gains,
                   policy_names=np.asarray(POLICIES[1:]), policy_move_scores=np.stack([directional, predicted]), ranked_candidate_index=a['candidate_index'][ranks])
        log(f'all-target {j + 1}/{len(panel["source_index"])} source={sid} backwards=34 elapsed={time.monotonic() - started:.1f}s')


def rank_agreement(fresh_rank, retained_rank, fresh_score, retained_score) -> dict:
    """Identity of the ranking plus two tolerant measures: Spearman correlation of move scores and top-100 overlap."""
    from ssc_geometry.statistics import spearman_rho
    return dict(ranking_identical=bool(np.array_equal(fresh_rank, retained_rank)),
                move_scores_bitwise_equal=bool(np.array_equal(fresh_score, retained_score)),
                move_score_max_abs_difference=float(np.max(np.abs(fresh_score - retained_score))),
                move_score_spearman=float(spearman_rho(fresh_score.astype(np.float64), retained_score.astype(np.float64))),
                top100_overlap=int(len(set(fresh_rank[:100].tolist()) & set(retained_rank[:100].tolist()))),
                first_rank_identical=bool(fresh_rank[0] == retained_rank[0]))


def compare(root: Path, out: Path, panel) -> dict:
    """Move scores and rankings of this run against data/search/ (contracts differ by construction)."""
    retained_scores = load_npz(root / paths.SEARCH_SCORES); retained_ranks = load_npz(root / paths.SEARCH_RANKS)
    retained_gradients = load_npz(root / paths.SEARCH_RUNNER_UP_GRADIENTS)
    offsets = retained_scores['source_offsets']; report = {'runner_up': {}, 'all_target': {}}
    for j, sid in enumerate(retained_scores['source_index']):
        sid = int(sid); lo, hi = offsets[j], offsets[j + 1]
        path = out / 'runner_up' / f'source_{sid:05d}.npz'
        if path.exists():
            z = load_npz(path)
            gs, ge = retained_gradients['gradient_offsets'][j:j + 2]
            report['runner_up'][sid] = dict(
                **rank_agreement(z['ranked_candidate_index'], retained_ranks['ranked_candidate_index'][0, lo:hi], z['move_score'], retained_scores['move_scores'][0, lo:hi]),
                gradient_max_abs_difference=float(np.max(np.abs(z['input_gradient'].ravel() - retained_gradients['runner_up_input_gradients'][gs:ge]))))
        path = out / 'all_target' / f'source_{sid:05d}.npz'
        if path.exists():
            z = load_npz(path)
            report['all_target'][sid] = {policy: rank_agreement(z['ranked_candidate_index'][k], retained_ranks['ranked_candidate_index'][k + 1, lo:hi],
                                                               z['policy_move_scores'][k], retained_scores['move_scores'][k + 1, lo:hi])
                                         for k, policy in enumerate(POLICIES[1:])}
    summary = {}
    for name, rows in report.items():
        entries = [v for r in rows.values() for v in (r.values() if name == 'all_target' else [r])]
        summary[name] = dict(sources=len(rows), rankings_identical=sum(e['ranking_identical'] for e in entries),
                             first_rank_identical=sum(e['first_rank_identical'] for e in entries), rankings=len(entries),
                             min_move_score_spearman=min((e['move_score_spearman'] for e in entries), default=None),
                             min_top100_overlap=min((e['top100_overlap'] for e in entries), default=None))
    write_json(out / 'comparison.json', dict(summary=summary, sources=report))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--step', choices=STEPS, nargs='+', required=True)
    p.add_argument('--h5', type=Path, help='official validation HDF5 (ssc_valid.h5)')
    p.add_argument('--panel', action='store_true', help='use the 100 released panel inputs instead of --h5 (panel-level steps only)')
    p.add_argument('--out', type=Path, required=True, help='output directory (created; reused to resume)')
    p.add_argument('--upstream', type=Path, default=ROOT / '.cache/upstream', help='pinned SpikeSCR source (scripts/prepare_upstream.py)')
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--sources', type=int, default=100, help='number of panel sources (smoke tests only; results are complete at 100)')
    a = p.parse_args()
    verify_integrity(ROOT)
    out = assert_output_safe(ROOT, a.out); out.mkdir(parents=True, exist_ok=True)
    panel = load_npz(ROOT / paths.PANEL_CLEAN_SCORES)
    panel = {k: v[:a.sources] for k, v in panel.items()}
    if a.step == ['compare']:
        print(json.dumps(compare(ROOT, out, panel), indent=2)); return
    if a.h5 is None and not a.panel:
        p.error('--h5 (full validation set) or --panel (released panel inputs) is required')
    import torch
    from ssc_geometry.upstream import load_model
    from ssc_geometry import forward
    from ssc_geometry.validation_data import ValidationData
    torch.set_num_threads(a.threads)
    data = ValidationData.from_panel(ROOT) if a.panel else ValidationData(ROOT, a.h5)
    model, device, _stages, environment = load_model(ROOT, a.upstream, a.device, allow_nonreference=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    config = read_json(ROOT / paths.EXPERIMENT_CONFIG)
    geometry = {k: v for k, v in load_npz(ROOT / paths.CANDIDATES).items()
                if k in ('source_index', 'candidate_index', 'from_bin', 'to_bin', 'feature', 'multiplicity', 'candidate_prediction', 'clean_prediction', 'label', 'transition_code')}
    payload = dict(package_version=__version__, script_sha256=sha256_file(Path(__file__)), checkpoint_sha256=config['checkpoint_sha256'],
                   state_dict_sha256=config['state_dict_sha256'], upstream_commit=config['upstream_commit'], data=data.provenance,
                   panel_clean_scores_sha256=sha256_file(ROOT / paths.PANEL_CLEAN_SCORES), candidates_sha256=sha256_file(ROOT / paths.CANDIDATES),
                   environment={k: v for k, v in environment.items() if k != 'upstream'}, threads=a.threads, score_atol=SCORE_ATOL, sources=a.sources,
                   surrogate='public ATan surrogate, LIF nodes in training mode for the backward only; batch normalisation and dropout in evaluation mode',
                   tie_break='ascending candidate_index', official_test_access=False)
    contract = json_hash(payload); contract_path = out / 'contract.json'
    if contract_path.exists() and read_json(contract_path)['contract_sha256'] != contract:
        raise ValueError('This output directory belongs to a different runtime/contract; choose a new one')
    write_json(contract_path, dict(contract_sha256=contract, payload=payload))
    started = time.monotonic()

    def log(message):
        print(f'[{time.monotonic() - started:8.1f}s] {message}', flush=True)
    gate_arrays = gate(model, forward, torch, data, panel, device, out, contract, log)
    for step in a.step:
        if step == 'runner-up': runner_up(model, forward, torch, data, panel, geometry, device, out, contract, log, gate_arrays)
        elif step == 'all-target': all_target(model, forward, torch, data, panel, geometry, device, out, contract, log, gate_arrays)
        elif step == 'compare': print(json.dumps(compare(ROOT, out, panel), indent=2))


if __name__ == '__main__':
    main()
