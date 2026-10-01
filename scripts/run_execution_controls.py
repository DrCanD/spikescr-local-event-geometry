"""Re-execute the execution controls of Appendix A on the frozen checkpoint (model inference; GPU for most steps).

Steps (``--step``), each resumable and each writing arrays with the same keys as the retained records
under data/execution/ so that ``compare`` can set the fresh run against them:

  clean-gate        100 panel sources: CPU singleton, GPU singleton, GPU singleton inside q/k isolation (bypass invariant)
  gpu-conditions    all 9,981 validation sources: native singleton, padding-matched singleton, batch 256, reversed batch 256
  cpu-singleton     all 9,981 validation sources as CPU singletons (Table 2, CPU row; Table A.1, first row)
  isolation         batch 256 and reversed batch 256 with q/k source isolation (Table A.1, rows 6-7; Appendix A.3)
  padding           valid-prefix / added-bin decomposition for the sources changed by padding plus 64 controls (Appendix A.2)
  pairs             the three [x,y] / [y,x] / [x,x] pair controls (Appendix A.3)
  candidate-replay  the 25,820 class-changing candidates on CPU and GPU (Table A.3, Appendix A.4)
  compare           label and score agreement between this run and the retained records

Inputs: the official validation HDF5 (``--h5``, hash-checked) and the pinned upstream source
(``scripts/prepare_upstream.py``). Nothing is written under the repository's data directories.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ssc_geometry import paths, __version__
from ssc_geometry.integrity import verify_integrity
from ssc_geometry.io import assert_output_safe, load_npz, read_json, write_json, write_csv, sha256_file
from ssc_geometry.core import move_one_count
from ssc_geometry.execution import transition, score_difference

STEPS = ('clean-gate', 'gpu-conditions', 'cpu-singleton', 'isolation', 'padding', 'pairs', 'candidate-replay', 'compare')
PANEL_REPLAYS = 10
PADDING_CONTROLS = 64


def json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_npz(path: Path, **arrays) -> None:
    temporary = path.with_name(path.name + '.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def read_part(path: Path, contract: str) -> dict:
    arrays = load_npz(path)
    if str(arrays.pop('contract_sha256')) != contract:
        raise ValueError(f'{path.name}: belongs to a different contract; use a new output directory')
    return arrays


def load_models(root: Path, upstream: Path, devices: list[str], threads: int):
    """One frozen model per device; the loader may run once per process, so devices share one import."""
    import torch
    from ssc_geometry.upstream import load_model
    from ssc_geometry import forward
    torch.set_num_threads(threads)
    models, environment = {}, None
    first = True
    for device in devices:
        if first:
            model, dev, _stages, environment = load_model(root, upstream, device, allow_nonreference=True)
            first = False
            base = model
        else:
            import copy
            model = copy.deepcopy(base).to(torch.device(device)); dev = torch.device(device)
            model.eval(); forward.reset_spiking_state(model)
        models[device] = (model, dev)
    environment = dict(environment); environment['threads'] = threads
    return models, environment, forward


def group_inputs(data, lo, hi):
    loaded = [data.get(sid) for sid in range(lo, hi)]
    lengths = np.array([len(x) for x, _ in loaded], dtype=np.int32); H = int(lengths.max())
    inputs = np.zeros((hi - lo, H, 140), dtype=np.float32)
    for j, (x, _) in enumerate(loaded):
        inputs[j, :len(x)] = x
    labels = np.array([y for _, y in loaded], dtype=np.int16)
    hashes = np.array([data.input_hash(x) for x, _ in loaded])
    return loaded, inputs, lengths, labels, hashes


# ---------------------------------------------------------------- steps
def clean_gate(models, forward, data, panel, out, contract, log):
    path = out / 'clean_gate.npz'
    if path.exists():
        z = read_part(path, contract)
    else:
        ids = panel['source_index']; cpu_scores, gpu_scores, isolated = [], [], []
        gpu_model, gpu = models['cuda']; cpu_model, cpu = models['cpu']
        for j, sid in enumerate(ids):
            x, _ = data.get(int(sid))
            cpu_scores.append(forward.batch_scores(cpu_model, x[None], [len(x)], cpu)[0])
            gpu_scores.append(forward.batch_scores(gpu_model, x[None], [len(x)], gpu)[0])
            with forward.qk_isolation(gpu_model):
                isolated.append(forward.batch_scores(gpu_model, x[None], [len(x)], gpu)[0])
            log(f'clean-gate {j + 1}/100')
        z = dict(source_index=ids, label=np.array([data.get(int(s))[1] for s in ids], dtype=np.int16), canonical_scores=panel['scores'],
                 CPU_scores=np.stack(cpu_scores), GPU_scores=np.stack(gpu_scores), GPU_qk_isolated_scores=np.stack(isolated))
        atomic_npz(path, contract_sha256=np.asarray(contract), **z)
    if not np.array_equal(z['GPU_scores'], z['GPU_qk_isolated_scores']):
        raise ValueError('q/k isolation changed singleton scores; the bypass invariant failed')
    canonical, gpu_pred, cpu_pred = z['canonical_scores'].argmax(1), z['GPU_scores'].argmax(1), z['CPU_scores'].argmax(1)
    summary = dict(sources=len(z['source_index']), CPU_GPU_clean_label_disagreements=int(np.sum(cpu_pred != gpu_pred)),
                   GPU_vs_canonical_clean_label_disagreements=int(np.sum(gpu_pred != canonical)),
                   CPU_vs_canonical_clean_label_disagreements=int(np.sum(cpu_pred != canonical)),
                   GPU_vs_canonical_max_abs_score=float(np.max(np.abs(z['GPU_scores'] - z['canonical_scores']))),
                   CPU_vs_canonical_max_abs_score=float(np.max(np.abs(z['CPU_scores'] - z['canonical_scores']))),
                   singleton_isolation_bitwise_equal=True)
    write_json(out / 'clean_gate.json', summary)
    return summary


def gpu_conditions(models, forward, data, panel, out, contract, log, limit):
    model, device = models['cuda']; parts = []; folder = out / 'gpu_conditions'; folder.mkdir(exist_ok=True)
    replay_ids = set(np.sort(panel['source_index'])[:PANEL_REPLAYS].tolist())
    for lo in range(0, limit, 256):
        hi = min(lo + 256, limit); path = folder / f'group_{lo:05d}_{hi:05d}.npz'
        if path.exists():
            rows = read_part(path, contract)
            if not np.array_equal(rows['source_index'], np.arange(lo, hi)):
                raise ValueError('Saved group identity differs')
        else:
            loaded, inputs, lengths, labels, hashes = group_inputs(data, lo, hi); H = inputs.shape[1]
            s256 = forward.batch_scores(model, inputs, lengths, device)
            srev = forward.batch_scores(model, np.array(inputs[::-1], copy=True), lengths[::-1], device)[::-1]
            native, padded, replayed = [], [], []
            for j, (x, _) in enumerate(loaded):
                native.append(forward.batch_scores(model, x[None], [len(x)], device)[0])
                padded.append(forward.batch_scores(model, inputs[j:j + 1], lengths[j:j + 1], device)[0])
                if lo + j in replay_ids:  # identical-input replays establish within-runtime repeatability
                    again = forward.batch_scores(model, x[None], [len(x)], device)[0]
                    if not np.array_equal(again, native[-1]):
                        raise RuntimeError('Identical GPU singleton replay changed scores')
                    replayed.append(lo + j)
            rows = dict(source_index=np.arange(lo, hi, dtype=np.int32), label=labels, native_horizon=lengths,
                        padded_horizon=np.full(hi - lo, H, dtype=np.int32), input_sha256=hashes,
                        B1_native_scores=np.stack(native), B1_padding_matched_scores=np.stack(padded), B256_scores=s256,
                        B256_reversed_scores=srev, replayed_source_index=np.array(replayed, dtype=np.int64))
            atomic_npz(path, contract_sha256=np.asarray(contract), **rows)
        parts.append(rows); log(f'gpu-conditions {hi}/{limit}')
    combined = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    for key in ('B1_native', 'B1_padding_matched', 'B256', 'B256_reversed'):
        combined[key + '_prediction'] = combined[key + '_scores'].argmax(1)
    atomic_npz(out / 'validation_gpu_conditions.npz', contract_sha256=np.asarray(contract), **combined)
    return combined


def cpu_singleton(models, forward, data, out, contract, log, limit):
    model, device = models['cpu']; parts = []; folder = out / 'cpu_singleton'; folder.mkdir(exist_ok=True)
    for lo in range(0, limit, 64):
        hi = min(lo + 64, limit); path = folder / f'records_{lo:05d}_{hi:05d}.npz'
        if path.exists():
            rows = read_part(path, contract)
        else:
            scores, labels, horizons, hashes = [], [], [], []
            for sid in range(lo, hi):
                x, y = data.get(sid)
                scores.append(forward.batch_scores(model, x[None], [len(x)], device)[0]); labels.append(y); horizons.append(len(x)); hashes.append(data.input_hash(x))
            rows = dict(source_index=np.arange(lo, hi, dtype=np.int32), label=np.array(labels, dtype=np.int16), scores=np.stack(scores),
                        prediction=np.stack(scores).argmax(1).astype(np.int16), horizon=np.array(horizons, dtype=np.int32), input_sha256=np.array(hashes))
            atomic_npz(path, contract_sha256=np.asarray(contract), **rows)
        parts.append(rows); log(f'cpu-singleton {hi}/{limit}')
    combined = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    atomic_npz(out / 'validation_cpu_singleton.npz', contract_sha256=np.asarray(contract), **combined)
    return combined


def isolation(models, forward, data, out, contract, log, limit):
    model, device = models['cuda']; parts = []; folder = out / 'isolation'; folder.mkdir(exist_ok=True)
    for lo in range(0, limit, 256):
        hi = min(lo + 256, limit); path = folder / f'group_{lo:05d}_{hi:05d}.npz'
        if path.exists():
            rows = read_part(path, contract)
        else:
            _loaded, inputs, lengths, labels, _hashes = group_inputs(data, lo, hi)
            rows = dict(source_index=np.arange(lo, hi, dtype=np.int32), label=labels, native_horizon=lengths, padded_horizon=np.asarray(inputs.shape[1]))
            rows['original_scores'] = forward.batch_scores(model, inputs, lengths, device)
            rows['original_reversed_scores'] = forward.batch_scores(model, inputs[::-1], lengths[::-1], device)[::-1]
            with forward.qk_isolation(model):
                rows['qk_isolated_scores'] = forward.batch_scores(model, inputs, lengths, device)
                rows['qk_isolated_reversed_scores'] = forward.batch_scores(model, inputs[::-1], lengths[::-1], device)[::-1]
            atomic_npz(path, contract_sha256=np.asarray(contract), **rows)
        parts.append(rows); log(f'isolation {hi}/{limit}')
    keys = ('source_index', 'label', 'original_scores', 'original_reversed_scores', 'qk_isolated_scores', 'qk_isolated_reversed_scores')
    combined = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    atomic_npz(out / 'qk_isolation.npz', contract_sha256=np.asarray(contract), **combined)
    predictions = {c: combined[c + '_scores'].argmax(1) for c in ('original', 'original_reversed', 'qk_isolated', 'qk_isolated_reversed')}
    summary = dict(sources=len(combined['label']), original_order_changed_labels=int(np.sum(predictions['original'] != predictions['original_reversed'])),
                   isolated_order_changed_labels=int(np.sum(predictions['qk_isolated'] != predictions['qk_isolated_reversed'])),
                   isolated_order_score_difference=score_difference(combined['qk_isolated_scores'], combined['qk_isolated_reversed_scores']),
                   conditions_correct={c: int(np.sum(p == combined['label'])) for c, p in predictions.items()},
                   layout_example=forward.lane_ownership(int(parts[0]['padded_horizon']), 256, 256))
    write_json(out / 'isolation.json', summary)
    return summary


def padding(models, forward, data, out, contract, log):
    model, device = models['cuda']; history = read_part(out / 'validation_gpu_conditions.npz', contract)
    gap = history['padded_horizon'] - history['native_horizon']
    changed = history['B1_native_prediction'] != history['B1_padding_matched_prediction']
    controls = np.flatnonzero((gap > 0) & ~changed)
    controls = controls[np.unique(np.linspace(0, len(controls) - 1, min(PADDING_CONTROLS, len(controls)), dtype=int))]
    ids = np.unique(np.concatenate([np.flatnonzero(changed), controls])).astype(np.int32)
    folder = out / 'padding'; folder.mkdir(exist_ok=True); parts = []
    for j, sid in enumerate(ids):
        path = folder / f'source_{sid:05d}.npz'
        if path.exists():
            row = read_part(path, contract)
        else:
            x, label = data.get(int(sid)); H = int(history['padded_horizon'][sid])
            padded_input = np.zeros((1, H, 140), dtype=np.float32); padded_input[0, :len(x)] = x
            native = forward.batch_scores(model, x[None], [len(x)], device)[0]
            full, prefix, extra = forward.batch_scores(model, padded_input, [len(x)], device, prefix=len(x))
            if np.max(np.abs(full.astype(float) - prefix.astype(float) - extra.astype(float))) > 1e-4:
                raise ValueError('Prefix/extra score decomposition failed')
            if abs(float(extra.sum()) - float(H - len(x))) > 1e-3:
                raise ValueError('Added softmax mass does not equal the number of added bins')
            row = dict(source_index=np.asarray(sid), label=np.asarray(label), native_horizon=np.asarray(len(x)), padded_horizon=np.asarray(H),
                       native_scores=native, padded_scores=full, padded_valid_prefix_scores=prefix, padded_extra_scores=extra)
            atomic_npz(path, contract_sha256=np.asarray(contract), **row)
        parts.append(row); log(f'padding {j + 1}/{len(ids)}')
    native = np.stack([r['native_scores'] for r in parts]); full = np.stack([r['padded_scores'] for r in parts])
    prefix = np.stack([r['padded_valid_prefix_scores'] for r in parts]); extra = np.stack([r['padded_extra_scores'] for r in parts])
    p_native, p_full, p_prefix, p_native_extra = native.argmax(1), full.argmax(1), prefix.argmax(1), (native + extra).argmax(1)
    original_changed = changed[ids]
    atomic_npz(out / 'padding_decomposition.npz', contract_sha256=np.asarray(contract), source_index=ids, native_scores=native, padded_scores=full,
               padded_valid_prefix_scores=prefix, padded_extra_scores=extra)
    summary = dict(diagnostic_sources=len(ids), selected_from_padding_changes=int(original_changed.sum()),
                   unchanged_positive_padding_controls=int((~original_changed).sum()), padding_label_changes=int(np.sum(p_native != p_full)),
                   prefix_only_labels_different_from_native=int(np.sum(p_prefix != p_native)),
                   padding_changes_restored_by_excluding_extra_bins=int(np.sum((p_full != p_native) & (p_prefix == p_native))),
                   native_plus_observed_extra_changes_label=int(np.sum(p_native_extra != p_native)),
                   native_label_disagreements_with_gpu_conditions=int(np.sum(p_native != history['B1_native_prediction'][ids])),
                   padded_label_disagreements_with_gpu_conditions=int(np.sum(p_full != history['B1_padding_matched_prediction'][ids])))
    write_json(out / 'padding_decomposition.json', summary)
    return summary


def pairs(models, forward, data, out, contract, log):
    model, device = models['cuda']; history = read_part(out / 'validation_gpu_conditions.npz', contract)
    changed = np.flatnonzero(history['B256_prediction'] != history['B256_reversed_prediction'])
    groups = sorted(set((changed // 256).tolist()))[:3]; selected = []
    for group in groups:
        x_id = int(changed[changed // 256 == group][0])
        y_id = group * 256 + ((x_id - group * 256 + 1) % min(256, 9981 - group * 256))
        selected.append((x_id, y_id, int(history['padded_horizon'][x_id])))
    rows = []
    for pair_no, (x_id, y_id, H) in enumerate(selected):
        x, _ = data.get(x_id); y, _ = data.get(y_id)
        inputs = np.zeros((2, H, 140), dtype=np.float32); inputs[0, :len(x)] = x; inputs[1, :len(y)] = y
        lengths = np.asarray([len(x), len(y)], dtype=np.int32)
        singleton = np.stack([forward.batch_scores(model, inputs[j:j + 1], lengths[j:j + 1], device)[0] for j in range(2)])
        for mode in ('original', 'q', 'k', 'qk'):
            context = forward.qk_isolation(model, mode) if mode != 'original' else _Null()
            with context:
                values = {name: forward.batch_scores(model, val, lens, device) for name, val, lens in
                          [('xy', inputs, lengths), ('yx', inputs[::-1], lengths[::-1]), ('xx', inputs[[0, 0]], lengths[[0, 0]]), ('xy_repeat', inputs, lengths)]}
            if not np.array_equal(values['xy'], values['xy_repeat']):
                raise ValueError('Identical-input pair replay changed scores')
            reverse = values['yx'][::-1]
            rows.append(dict(pair=pair_no, x_source=x_id, y_source=y_id, padded_horizon=H, mode=mode,
                             order_changed_labels=int(np.sum(values['xy'].argmax(1) != reverse.argmax(1))),
                             order_score_max_abs=float(np.max(np.abs(values['xy'] - reverse))),
                             duplicate_x_position_score_max_abs=float(np.max(np.abs(values['xx'][0] - values['xx'][1]))),
                             versus_padded_singleton_max_abs=float(np.max(np.abs(values['xy'] - singleton)))))
        log(f'pairs {pair_no + 1}/{len(selected)}')
    write_csv(out / 'pair_controls.csv', rows)
    write_json(out / 'pair_controls.json', dict(pairs=rows, structural=[dict(pair=i, source_ids=[a, b], layout=forward.lane_ownership(H, 2, 256)) for i, (a, b, H) in enumerate(selected)]))
    return rows


class _Null:
    def __enter__(self): return self
    def __exit__(self, *exc): return False


def candidate_replay(models, forward, data, root, out, contract, log, candidate_limit=25820):
    geometry = load_npz(root / paths.CANDIDATES); clean = read_part(out / 'clean_gate.npz', contract)
    selected = np.flatnonzero(geometry['transition_code'] != 0)
    if len(selected) != 25820:
        raise ValueError('Expected exactly 25,820 class-changing candidates')
    selected = selected[:candidate_limit]
    folder = out / 'candidate_replay'; folder.mkdir(exist_ok=True); parts = []; done = 0
    gpu_model, gpu = models['cuda']; cpu_model, cpu = models['cpu']
    for sid in np.unique(geometry['source_index'][selected]):
        indices = selected[geometry['source_index'][selected] == sid]; x, _ = data.get(int(sid))
        for lo in range(0, len(indices), 128):
            rows = indices[lo:lo + 128]; path = folder / f'source_{sid:05d}_chunk_{lo:05d}.npz'
            if path.exists():
                part = read_part(path, contract)
            else:
                cpu_scores, gpu_scores = [], []
                for row_index in rows:
                    candidate = move_one_count(x, int(geometry['from_bin'][row_index]), int(geometry['to_bin'][row_index]), int(geometry['feature'][row_index]))
                    cpu_scores.append(forward.batch_scores(cpu_model, candidate[None], [len(x)], cpu)[0])
                    gpu_scores.append(forward.batch_scores(gpu_model, candidate[None], [len(x)], gpu)[0])
                part = dict(geometry_row_index=rows.astype(np.int32), source_index=geometry['source_index'][rows], candidate_index=geometry['candidate_index'][rows],
                            label=geometry['label'][rows], from_bin=geometry['from_bin'][rows], to_bin=geometry['to_bin'][rows], feature=geometry['feature'][rows],
                            canonical_candidate_prediction=geometry['candidate_prediction'][rows], canonical_transition_code=geometry['transition_code'][rows],
                            CPU_scores=np.stack(cpu_scores), GPU_scores=np.stack(gpu_scores))
                atomic_npz(path, contract_sha256=np.asarray(contract), **part)
            parts.append(part); done += len(rows); log(f'candidate-replay {done}/{len(selected)}')
    combined = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    order = np.argsort(combined['geometry_row_index']); combined = {k: v[order] for k, v in combined.items()}
    lookup = {int(s): j for j, s in enumerate(clean['source_index'])}
    for device in ('CPU', 'GPU'):
        clean_pred = np.array([clean[f'{device}_scores'][lookup[int(s)]].argmax() for s in combined['source_index']])
        pred = combined[f'{device}_scores'].argmax(1)
        combined[f'{device}_prediction'] = pred; combined[f'{device}_clean_prediction'] = clean_pred
        combined[f'{device}_transition_code'] = transition(combined['label'], clean_pred, pred)
    atomic_npz(out / 'candidate_replay.npz', contract_sha256=np.asarray(contract), **combined)
    return dict(candidates=len(order), CPU_GPU_label_agreements=int(np.sum(combined['CPU_prediction'] == combined['GPU_prediction'])))


def compare(root: Path, out: Path) -> dict:
    """Labels and scores of this run against the retained records (the contract hashes differ by construction)."""
    report = {}
    for name, relative, score_keys in [('validation_gpu_conditions', paths.EXECUTION_GPU_CONDITIONS, ('B1_native', 'B1_padding_matched', 'B256', 'B256_reversed')),
                                       ('validation_cpu_singleton', paths.EXECUTION_CPU_SINGLETON, ('',)),
                                       ('qk_isolation', paths.EXECUTION_QK_ISOLATION, ('original', 'original_reversed', 'qk_isolated', 'qk_isolated_reversed')),
                                       ('candidate_replay', paths.EXECUTION_CANDIDATE_REPLAY, ('CPU', 'GPU'))]:
        path = out / f'{name}.npz'
        if not path.exists():
            report[name] = 'not run'; continue
        fresh, retained = load_npz(path), load_npz(root / relative); entry = {}
        for key in score_keys:
            k = (key + '_scores') if key else 'scores'
            n = min(len(fresh[k]), len(retained[k]))
            entry[k] = dict(sources=n, label_disagreements=int(np.sum(fresh[k][:n].argmax(1) != retained[k][:n].argmax(1))),
                            max_abs_score_difference=float(np.max(np.abs(fresh[k][:n] - retained[k][:n]))))
        report[name] = entry
    write_json(out / 'comparison.json', report)
    return report


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--step', choices=STEPS, nargs='+', required=True)
    p.add_argument('--h5', type=Path, help='official validation HDF5 (ssc_valid.h5)')
    p.add_argument('--panel', action='store_true', help='use the 100 released panel inputs instead of --h5 (panel-level steps only)')
    p.add_argument('--out', type=Path, required=True, help='output directory (created; reused to resume)')
    p.add_argument('--upstream', type=Path, default=ROOT / '.cache/upstream', help='pinned SpikeSCR source (scripts/prepare_upstream.py)')
    p.add_argument('--gpu', default='cuda', help='CUDA device for the GPU conditions')
    p.add_argument('--threads', type=int, default=1, help='CPU threads for the CPU singleton runs')
    p.add_argument('--limit', type=int, default=9981, help='number of validation sources (smoke tests only; results are complete at 9981)')
    p.add_argument('--candidate-limit', type=int, default=25820, help='number of class-changing candidates to replay (smoke tests only)')
    a = p.parse_args()
    verify_integrity(ROOT)
    out = assert_output_safe(ROOT, a.out); out.mkdir(parents=True, exist_ok=True)
    if a.step == ['compare']:
        print(json.dumps(compare(ROOT, out), indent=2)); return
    if a.h5 is None and not a.panel:
        p.error('--h5 (full validation set) or --panel (released panel inputs) is required')
    from ssc_geometry.validation_data import ValidationData
    data = ValidationData.from_panel(ROOT) if a.panel else ValidationData(ROOT, a.h5)
    panel = load_npz(ROOT / paths.PANEL_CLEAN_SCORES)
    config = read_json(ROOT / paths.EXPERIMENT_CONFIG)
    need_gpu = any(s in a.step for s in ('clean-gate', 'gpu-conditions', 'isolation', 'padding', 'pairs', 'candidate-replay'))
    need_cpu = any(s in a.step for s in ('clean-gate', 'cpu-singleton', 'candidate-replay'))
    devices = list(dict.fromkeys(([a.gpu] if need_gpu else []) + (['cpu'] if need_cpu else [])))
    models, environment, forward = load_models(ROOT, a.upstream, devices, a.threads)
    if a.gpu in models:
        models['cuda'] = models[a.gpu]
    if need_gpu and a.gpu == 'cpu':
        print('WARNING: GPU conditions are being run on the CPU; this is a smoke test, not a result', flush=True)
    payload = dict(package_version=__version__, script_sha256=sha256_file(Path(__file__)), checkpoint_sha256=config['checkpoint_sha256'],
                   state_dict_sha256=config['state_dict_sha256'], upstream_commit=config['upstream_commit'], data=data.provenance,
                   panel_clean_scores_sha256=sha256_file(ROOT / paths.PANEL_CLEAN_SCORES), candidates_sha256=sha256_file(ROOT / paths.CANDIDATES),
                   environment=environment, threads=a.threads, sources=a.limit, candidates=a.candidate_limit, official_test_access=False,
                   scores='public readout: sum over every padded time bin of softmax(logits); no new masking of output scores',
                   batch_groups='ascending source indices, contiguous groups of 256; the last group keeps its native size',
                   execution='frozen model, float32, eval, state reset before every forward; AMP and TF32 off; deterministic algorithms')
    contract = json_hash(payload); contract_path = out / 'contract.json'
    if contract_path.exists() and read_json(contract_path)['contract_sha256'] != contract:
        raise ValueError('This output directory belongs to a different runtime/contract; choose a new one')
    write_json(contract_path, dict(contract_sha256=contract, payload=payload))
    started = time.monotonic()

    def log(message):
        print(f'[{time.monotonic() - started:8.1f}s] {message}', flush=True)

    results = {}
    for step in a.step:
        if step == 'clean-gate': results[step] = clean_gate(models, forward, data, panel, out, contract, log)
        elif step == 'gpu-conditions': gpu_conditions(models, forward, data, panel, out, contract, log, a.limit); results[step] = 'validation_gpu_conditions.npz'
        elif step == 'cpu-singleton': cpu_singleton(models, forward, data, out, contract, log, a.limit); results[step] = 'validation_cpu_singleton.npz'
        elif step == 'isolation': results[step] = isolation(models, forward, data, out, contract, log, a.limit)
        elif step == 'padding': results[step] = padding(models, forward, data, out, contract, log)
        elif step == 'pairs': results[step] = pairs(models, forward, data, out, contract, log)
        elif step == 'candidate-replay': results[step] = candidate_replay(models, forward, data, ROOT, out, contract, log, a.candidate_limit)
        elif step == 'compare': results[step] = compare(ROOT, out)
    write_json(out / 'summary.json', dict(contract_sha256=contract, steps=results, elapsed_seconds=time.monotonic() - started, official_test_access=False))
    print(json.dumps(results, indent=2, default=str))


if __name__ == '__main__':
    main()
