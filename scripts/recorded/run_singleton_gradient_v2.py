#!/usr/bin/env python3
"""Frozen SpikeSCR validation-only B=1 evaluation and one-step gradient ranking.

No training, optimizer, official-test discovery, network access or dependency
installation is performed. Scientific runs fail closed on checkpoint/source,
input provenance and the 100-source canonical score/prediction parity gates.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import numpy as np

VERSION = 'singleton-validation-gradient-20260929-v2'
COMMIT = '095f418f53b3b24c21caf558225c65ad674d44b1'
CHECKPOINT_SHA = '8ce4d237f8c1179db46f6e96028f5dd84400b0db441b0e086a4a8195c121234d'
STATE_SHA = 'b1651d836fbd76070ad6bcac863cd870904c8e65e1a8e5e18a2f0c30e25ec68c'
VALID_SHA = '6365c36ff8c680c4b0caf27ca52faa59e14359db5810b528f40dd80e2bcb9210'
INTERNAL_CONTRACT = 'ca4efc03802f1de36e4ffcddde165e75b600111187c57527bcc39380448e4404'
EB18_CONTRACT = 'ded89cf6e33d61ab4ae6d24a51f920a69e9a9cef0e9ae88741a2f91edabeaa6e'
PINNED = {
    'models/spikescr.py': '33cc3e2d68ae8e32f408e9357c3f822706294dd8',
    'module/spikscr_module.py': '1d212210378c5c8124f5e6c1f5a7bdbdae1677d1',
    'module/conv.py': '632d6319924a8ca9a0fbe67e7aa11b2467e46b46',
    'configs/best_config_SSC_former.py': '9e0d97705eefa6060cb7f400360e826634cd5ae8',
}
DEPS = {'spikingjelly': '0.0.0.0.14', 'einops': '0.8.0', 'rotary-embedding-torch': '0.8.4'}


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    os.replace(tmp, path)


def atomic_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def exact_transform(times, units):
    """Literal EB18 exact_event_transform semantics, including last-bin clamp."""
    times = np.asarray(times, dtype=np.float64)
    units = np.asarray(units, dtype=np.int64)
    if len(times) != len(units) or np.any(np.diff(times) < 0):
        raise ValueError('Invalid validation event times/units')
    if len(units) and (units.min() < 0 or units.max() >= 700):
        raise ValueError('Invalid raw input channel')
    if not len(times):
        return np.zeros((1, 140), dtype=np.uint16)
    ms = (times-times[0])*1000.0
    n = max(1, int(math.ceil(float(ms[-1])/5.0)))
    frame = np.minimum(np.floor_divide(ms, 5).astype(np.int64), n-1)
    out = np.bincount(frame*140+units//5, minlength=n*140).reshape(n, 140)
    if out.max(initial=0) > 65535:
        raise ValueError('Count overflow')
    return out.astype(np.uint16)


class ValidationData:
    def __init__(self, path, manifest_path=None):
        self.path = Path(path); self.cached_shard = None; self.shard_data = None
        if self.path.is_file() and self.path.suffix in ('.h5', '.hdf5'):
            if file_hash(self.path) != VALID_SHA:
                raise ValueError('HDF5 is not the frozen SSC validation file; no other split is accepted')
            import h5py
            self.h5 = h5py.File(self.path, 'r'); self.kind = 'h5'
            self.labels = np.asarray(self.h5['labels'], dtype=np.int64)
            self.ids = np.arange(len(self.labels), dtype=np.int64)
            self.provenance = {'kind': 'h5', 'source_sha256': VALID_SHA}
        else:
            manifest = Path(manifest_path) if manifest_path else self.path/'manifest.json'
            m = json.loads(manifest.read_text())
            if m.get('source_sha256') != VALID_SHA or m.get('split') != 'valid':
                raise ValueError('Only a manifest bound to the frozen validation HDF5 is accepted')
            if not m.get('completed') or m.get('smoke') or int(m.get('samples', -1)) != 9981:
                raise ValueError('Validation manifest is not a complete 9,981-record cache')
            if (m.get('time_step_ms'), m.get('spatial_bin'), m.get('features')) != (5, 5, 140):
                raise ValueError('Validation preprocessing contract mismatch')
            self.provenance = {'kind': 'cache', 'source_sha256': VALID_SHA,
                               'manifest_sha256': file_hash(manifest), 'data_fingerprint': m['data_fingerprint']}
            if (self.path/'events.npy').is_file():
                self.kind = 'stage'; stage = json.loads((self.path/'stage_manifest.json').read_text())
                if stage['data_fingerprint'] != m['data_fingerprint']:
                    raise ValueError('Stage/persistent manifest identity differs')
                for name in ('events', 'lengths', 'labels', 'source_indices'):
                    p = self.path/(name+'.npy')
                    if file_hash(p) != stage['files'][name]['sha256']:
                        raise ValueError('Staged input hash mismatch: '+name)
                self.events = np.load(self.path/'events.npy', mmap_mode='r')
                self.lengths = np.load(self.path/'lengths.npy', mmap_mode='r')
                self.labels = np.load(self.path/'labels.npy', mmap_mode='r')
                self.ids = np.load(self.path/'source_indices.npy', mmap_mode='r')
            else:
                self.kind = 'shards'; self.rows = m['shards']; self.lookup = {}
                self.labels = np.empty(9981, dtype=np.int64); self.ids = np.arange(9981)
                seen = set()
                for j, row in enumerate(self.rows):
                    p = self.path/'shards'/row['filename']
                    if file_hash(p) != row['sha256']:
                        raise ValueError('Persistent shard hash mismatch: '+str(p))
                    with np.load(p, allow_pickle=False) as z:
                        ids = z['source_indices'].astype(np.int64)
                        if len(ids) != int(row['stop'])-int(row['start']):
                            raise ValueError('Shard length mismatch')
                        for k, sid in enumerate(ids):
                            if int(sid) in seen or not 0 <= sid < 9981:
                                raise ValueError('Repeated/invalid validation source')
                            seen.add(int(sid)); self.lookup[int(sid)] = (j, k)
                        self.labels[ids] = z['labels']
                if seen != set(range(9981)):
                    raise ValueError('Incomplete validation shards')
        if len(self.labels) != 9981 or not np.array_equal(self.ids, np.arange(9981)):
            raise ValueError('Validation must contain sources 0..9980 exactly once')
        if np.any((self.labels < 0)|(self.labels >= 35)):
            raise ValueError('Invalid validation labels')

    def get(self, sid):
        sid = int(sid)
        if self.kind == 'h5':
            x = exact_transform(self.h5['spikes/times'][sid], self.h5['spikes/units'][sid])
        elif self.kind == 'stage':
            x = self.events[sid, :int(self.lengths[sid])]
        else:
            j, k = self.lookup[sid]
            if self.cached_shard != j:
                with np.load(self.path/'shards'/self.rows[j]['filename'], allow_pickle=False) as z:
                    self.shard_data = {n: z[n] for n in ('events', 'lengths')}
                self.cached_shard = j
            x = self.shard_data['events'][k, :int(self.shard_data['lengths'][k])]
        x = np.array(x, dtype=np.float32, copy=True)
        if x.ndim != 2 or x.shape[1] != 140 or len(x) < 1 or np.any(x < 0):
            raise ValueError('Invalid transformed validation input')
        return x, int(self.labels[sid])


def load_panel(path):
    """Compact refs or the existing per-source EB15R2 canonical artifacts."""
    p = Path(path); refs = {}
    if p.is_file():
        with np.load(p, allow_pickle=False) as z:
            if str(z['parent_contract_sha256']) != EB18_CONTRACT:
                raise ValueError('Compact panel parent must be EB18')
            for j, sid in enumerate(z['source_index']):
                refs[int(sid)] = {'label': int(z['label'][j]), 'prediction': int(z['clean_prediction'][j]),
                                  'scores': z['clean_scores'][j].astype(np.float32)}
    else:
        for f in sorted(p.rglob('exact_internal_complete.npz')):
            with np.load(f, allow_pickle=False) as z:
                if str(z['contract_sha256']) != INTERNAL_CONTRACT or str(z['parent_contract_sha256']) != EB18_CONTRACT:
                    raise ValueError('Panel source contract differs from canonical EB15R2/EB18')
                sid = int(z['source_index'])
                if sid in refs:
                    raise ValueError('Duplicate panel source')
                refs[sid] = {'label': int(z['label']), 'prediction': int(z['clean_prediction']),
                             'scores': z['clean_scores'].astype(np.float32)}
    if len(refs) != 100 or any(r['scores'].shape != (35,) for r in refs.values()):
        raise ValueError('Exactly 100 canonical clean references are required')
    h = hashlib.sha256()
    for sid in sorted(refs):
        h.update(np.asarray([sid, refs[sid]['label'], refs[sid]['prediction']], dtype=np.int64).tobytes())
        h.update(refs[sid]['scores'].tobytes())
    return refs, h.hexdigest()


def build_model(repo, checkpoint, device, threads):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    for name, expected in DEPS.items():
        if importlib.metadata.version(name) != expected:
            raise ValueError('Pinned dependency version mismatch: '+name)
    repo = Path(repo)
    for name, expected in PINNED.items():
        b = (repo/name).read_bytes()
        actual = hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
        if actual != expected:
            raise ValueError('Pinned source hash mismatch: '+name)
    if file_hash(checkpoint) != CHECKPOINT_SHA:
        raise ValueError('Only the frozen epoch-282 checkpoint is accepted')
    random.seed(312); np.random.seed(312); torch.manual_seed(312)
    torch.set_num_threads(threads)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(312)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    sys.path.insert(0, str(repo.resolve()))
    from spikingjelly.activation_based import layer, functional, neuron
    original_bn = layer.BatchNorm1d
    class BatchNorm1d3DAdapter(original_bn):
        def forward(self, x):
            if getattr(self, 'step_mode', 's') == 'm' and x.ndim == 3:
                return super().forward(x.unsqueeze(-1)).squeeze(-1)
            return super().forward(x)
    layer.BatchNorm1d = BatchNorm1d3DAdapter
    config = importlib.import_module('configs.best_config_SSC_former').Config()
    assert not hasattr(config, 'use_ln')
    config.use_ln = False; config.time_step = 5; config.epochs = 300; config.seed = 312
    config.backend = 'torch'; config.n_hidden_neurons = int(config.n_hidden_neurons_list[0])
    config.hidden_dims = config.mlp_ratio*config.n_hidden_neurons
    config.n_inputs = 140; config.n_outputs = 35
    model = importlib.import_module('models.spikescr').SpikeDrivenTransformer(config).to(device)
    blob = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if int(blob['epoch'])+1 != 282: raise ValueError('Checkpoint epoch differs')
    h = hashlib.sha256()
    for name in sorted(blob['model_state_dict']):
        t = blob['model_state_dict'][name].detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(t.dtype).encode())
        h.update(np.asarray(t.shape, dtype=np.int64).tobytes()); h.update(t.numpy().tobytes())
    if h.hexdigest() != STATE_SHA: raise ValueError('Checkpoint tensor digest differs')
    model.load_state_dict(blob['model_state_dict'], strict=True)
    if sum(p.numel() for p in model.parameters()) != 3302416 or sum(p.numel() for p in model.parameters() if p.requires_grad) != 3302400:
        raise ValueError('Unexpected model parameter count')
    for p in model.parameters(): p.requires_grad_(False)
    model.eval(); functional.reset_net(model)
    env = {'torch': torch.__version__, 'device': str(device), 'threads': threads,
           'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
           'device_name': torch.cuda.get_device_name(device) if str(device).startswith('cuda') else 'CPU',
           'dependencies': DEPS}
    return model, torch, functional, neuron, env


def install_spike_hash_hooks(model):
    hashes = {}; handles = []
    def make_hook(name):
        def hook(module, inputs, output):
            value = output.detach().cpu().contiguous().numpy()
            hashes[name] = hashlib.sha256(value.tobytes()).hexdigest()
        return hook
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'LIFNode':
            handles.append(module.register_forward_hook(make_hook(name)))
    return hashes, handles


def forward_scores(model, torch, functional, x, device, capture_lif=False):
    functional.reset_net(model)
    hashes, handles = install_spike_hash_hooks(model) if capture_lif else ({}, [])
    inp = torch.from_numpy(x[None]).to(device)
    mask = torch.ones((1, len(x)), dtype=torch.bool, device=device)
    try:
        with torch.no_grad():
            s = torch.softmax(model(inp, mask).float(), dim=2).sum(dim=0)[0]
        scores = s.cpu().numpy().astype(np.float32)
        return (scores, hashes) if capture_lif else scores
    finally:
        for handle in handles: handle.remove()
        functional.reset_net(model)


def canonical_gate(model, torch, functional, data, refs, device, out, atol, contract, gate_limit=100):
    basename = 'canonical_panel_gate' if gate_limit == 100 else f'canonical_panel_probe_{gate_limit}'
    path = out/(basename+'.npz')
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            if str(z['contract_sha256']) == contract and len(z['source_index']) == 100 and bool(z['pass']):
                print('[GATE] reused contract-bound successful 100-source gate', flush=True)
                return
    ids = sorted(refs)[:gate_limit]; scores = []; errors = []; class_ok = []; started = time.monotonic(); spike_hash_rows = []; neuron_names = None
    for j, sid in enumerate(ids):
        x, label = data.get(sid); ref = refs[sid]
        if label != ref['label']: raise ValueError('Panel input label mismatch')
        s, spike_hashes = forward_scores(model, torch, functional, x, device, capture_lif=True)
        if neuron_names is None: neuron_names = sorted(spike_hashes)
        if sorted(spike_hashes) != neuron_names: raise ValueError('LIF hook topology changed')
        spike_hash_rows.append([spike_hashes[name] for name in neuron_names])
        if not np.isfinite(s).all(): raise ValueError('Nonfinite clean score')
        scores.append(s); errors.append(float(np.max(np.abs(s-ref['scores']))))
        class_ok.append(int(s.argmax()) == ref['prediction'])
        if (j+1)%10 == 0 or gate_limit < 10:
            print(f'[GATE] {j+1}/{len(ids)} max_abs={max(errors):.9g} class_mismatch={len(class_ok)-sum(class_ok)} elapsed={time.monotonic()-started:.2f}s', flush=True)
    passed = all(class_ok) and max(errors) <= atol
    atomic_npz(path, contract_sha256=np.asarray(contract), source_index=np.asarray(ids),
               scores=np.asarray(scores), max_abs_error=np.asarray(errors), class_parity=np.asarray(class_ok),
               lif_names=np.asarray(neuron_names), lif_spike_sha256=np.asarray(spike_hash_rows),
               **{'pass': np.asarray(passed)})
    atomic_json(out/(basename+'.json'), {'contract_sha256': contract, 'sources': len(ids),
                'completed_full_gate': len(ids) == 100, 'elapsed_seconds':time.monotonic()-started,
                'score_atol': atol, 'max_abs_error': max(errors), 'class_disagreements': len(ids)-sum(class_ok),
                'bit_exact_sources': sum(e == 0 for e in errors), 'pass': bool(passed)})
    if not passed:
        raise RuntimeError('Canonical 100-source parity failed. No validation or gradient inference proceeds; inspect gate output.')


def full_validation(model, torch, functional, data, device, out, contract, limit, max_seconds, start=0, stop=9981):
    root = out/'validation'; root.mkdir(exist_ok=True)
    started = time.monotonic(); stop = min(stop, start+limit, 9981)
    parts = []; computed = 0
    for lo in range(start, stop, 64):
        hi = min(lo+64, stop); p = root/f'records_{lo:05d}_{hi:05d}.npz'
        if p.exists():
            with np.load(p, allow_pickle=False) as z:
                if str(z['contract_sha256']) != contract or not np.array_equal(z['source_index'], np.arange(lo, hi)):
                    raise ValueError('Saved validation chunk has wrong identity')
                arrays = {k: z[k] for k in ('source_index','label','prediction','scores','horizon','input_sha256')}
        else:
            rows = []
            for sid in range(lo, hi):
                x, label = data.get(sid)
                s = forward_scores(model, torch, functional, x, device)
                if not np.isfinite(s).all(): raise ValueError('Nonfinite validation score')
                rows.append((sid, label, int(s.argmax()), s, len(x), hashlib.sha256(x.tobytes()).hexdigest()))
            arrays = {'source_index': np.asarray([r[0] for r in rows], dtype=np.int32),
                      'label': np.asarray([r[1] for r in rows], dtype=np.int16),
                      'prediction': np.asarray([r[2] for r in rows], dtype=np.int16),
                      'scores': np.asarray([r[3] for r in rows], dtype=np.float32),
                      'horizon': np.asarray([r[4] for r in rows], dtype=np.int32),
                      'input_sha256': np.asarray([r[5] for r in rows])}
            atomic_npz(p, contract_sha256=np.asarray(contract), **arrays); computed += hi-lo
        parts.append(arrays)
        correct = sum(int(np.sum(a['label'] == a['prediction'])) for a in parts)
        n = sum(len(a['label']) for a in parts)
        summary = {'contract_sha256': contract, 'completed': n == 9981, 'samples': n,
                   'requested_source_start':start,'requested_source_stop':stop,'requested_range_completed':n==stop-start,
                   'correct': correct, 'accuracy': correct/n, 'semantic_batch_size': 1,
                   'computational_batch_size': 1, 'new_forward_passes_this_invocation': computed,
                   'elapsed_seconds_this_invocation': time.monotonic()-started, 'official_test_access': False}
        atomic_json(root/'summary.json', summary)
        print(f'[VALID] {n}/9981 correct={correct} accuracy={correct/n:.8%}', flush=True)
        if max_seconds and time.monotonic()-started >= max_seconds: break
    if parts:
        atomic_npz(root/'predictions.npz', contract_sha256=np.asarray(contract),
                   **{k: np.concatenate([a[k] for a in parts]) for k in parts[0]})


def gradient_ranking(model, torch, functional, neuron, data, refs, device, geometry, out, contract, atol, budgets, limit, max_seconds):
    with np.load(out/'canonical_panel_gate.npz',allow_pickle=False) as z:
        if not bool(z['pass']) or len(z['source_index'])!=100 or str(z['contract_sha256'])!=contract:
            raise ValueError('A complete current-runtime canonical panel gate is required')
        runtime_clean={int(sid):z['scores'][j] for j,sid in enumerate(z['source_index'])}
        lif_names=list(map(str,z['lif_names']))
        runtime_spikes={int(sid):z['lif_spike_sha256'][j] for j,sid in enumerate(z['source_index'])}
    with np.load(geometry, allow_pickle=False) as z:
        g = {k: z[k] for k in ('source_index','candidate_index','from_bin','to_bin','feature','multiplicity',
                               'candidate_prediction','clean_prediction','label','transition_code')}
    if len(g['source_index']) != 725070 or set(map(int, np.unique(g['source_index']))) != set(refs):
        raise ValueError('Candidate map does not match the frozen 100-source panel')
    root = out/'gradient'; root.mkdir(exist_ok=True); started = time.monotonic(); results = []; gate_rows = []; new_backward=0
    for sid in sorted(refs)[:min(limit,100)]:
        p = root/f'source_{sid:05d}.npz'; rowpath = root/f'source_{sid:05d}.json'
        if p.exists() and rowpath.exists():
            row = json.loads(rowpath.read_text())
            if row.get('contract_sha256') != contract or not row.get('surrogate_forward_gate_pass'):
                raise ValueError('Saved gradient source has wrong identity')
            results.extend(row['budgets']); gate_rows.append(row['gate'])
            continue
        x, label = data.get(sid); ref = refs[sid]; mask = g['source_index'] == sid
        a = {k: v[mask] for k,v in g.items()}; order0 = np.argsort(a['candidate_index'], kind='stable')
        a = {k: v[order0] for k,v in a.items()}; n = len(a['candidate_index'])
        frm, to, feat = (a[k].astype(np.int64) for k in ('from_bin','to_bin','feature'))
        expected_n = 2*np.count_nonzero(x)-np.count_nonzero(x[0])-np.count_nonzero(x[-1])
        if n != expected_n or not np.array_equal(a['candidate_index'],np.arange(n)):
            raise ValueError('Candidate map coverage differs from validation input')
        if np.any(np.abs(to-frm)!=1) or np.any(frm<0) or np.any(to<0) or np.any(frm>=len(x)) or np.any(to>=len(x)) or np.any(feat<0) or np.any(feat>=140):
            raise ValueError('Illegal candidate coordinates')
        if np.any(x[frm,feat] <= 0) or not np.array_equal(x[frm,feat],a['multiplicity']):
            raise ValueError('Candidate multiplicities differ from validation input')
        if len(np.unique(np.stack([frm,to,feat],axis=1),axis=0)) != n:
            raise ValueError('Duplicate candidate move')
        if np.any(a['label']!=label) or np.any(a['clean_prediction']!=ref['prediction']):
            raise ValueError('Map clean labels/predictions differ')
        pred=a['candidate_prediction']; clean=ref['prediction']
        expected_code=np.zeros(n,dtype=np.uint8)
        changed=pred!=clean
        if clean==label: expected_code[changed]=1
        else:
            expected_code[changed & (pred==label)]=2
            expected_code[changed & (pred!=label)]=3
        if not np.array_equal(expected_code,a['transition_code']):
            raise ValueError('Stored transition codes do not match immutable predictions')
        # Surrogate path only: global model, BN and all dropout layers stay eval.
        model.eval(); functional.reset_net(model)
        lifs = [m for m in model.modules() if isinstance(m, neuron.LIFNode)]
        if not lifs: raise ValueError('No LIF nodes found')
        for m in lifs: m.train(True)
        if any(m.training for m in model.modules() if isinstance(m,(torch.nn.modules.batchnorm._BatchNorm,torch.nn.Dropout))):
            raise ValueError('BN or dropout was accidentally enabled')
        inp = torch.from_numpy(x[None]).to(device).requires_grad_(True)
        attnmask = torch.ones((1,len(x)),dtype=torch.bool,device=device)
        spike_hashes, handles = install_spike_hash_hooks(model)
        try:
            with torch.enable_grad():
                scores = torch.softmax(model(inp,attnmask).float(),dim=2).sum(dim=0)[0]
                s = scores.detach().cpu().numpy().astype(np.float32)
                err = float(np.max(np.abs(s-ref['scores'])))
                runtime_err=float(np.max(np.abs(s-runtime_clean[sid])))
                if sorted(spike_hashes)!=lif_names: raise ValueError('Gradient LIF topology differs')
                differing_lifs=[name for j,name in enumerate(lif_names) if spike_hashes[name]!=runtime_spikes[sid][j]]
                passed = int(s.argmax()) == ref['prediction'] and err <= atol and runtime_err <= atol and not differing_lifs
                gate = {'source_index': sid, 'max_abs_score_error': err,
                        'max_abs_to_current_eval_forward':runtime_err,
                        'bit_exact_to_current_eval_forward':bool(np.array_equal(s,runtime_clean[sid])),
                        'lif_spike_bit_parity':not differing_lifs,'differing_lif_modules':differing_lifs,
                        'class_parity': int(s.argmax())==ref['prediction'], 'pass': bool(passed)}
                if not passed:
                    atomic_json(root/f'source_{sid:05d}_FAILED_GATE.json',gate)
                    raise RuntimeError('Surrogate-mode hard-forward parity failed; no gradient rank is accepted')
                others = ref['scores'].copy(); others[ref['prediction']] = -np.inf
                rival = int(others.argmax())
                objective = scores[rival]-scores[ref['prediction']]
                grad = torch.autograd.grad(objective,inp)[0][0].detach().cpu().numpy().astype(np.float32)
                new_backward+=1
        finally:
            for handle in handles: handle.remove()
            model.eval(); functional.reset_net(model)
        if not np.isfinite(grad).all(): raise ValueError('Nonfinite input surrogate gradient')
        move_score = grad[to,feat]-grad[frm,feat]
        ranking = np.lexsort((a['candidate_index'],-move_score))
        candidate_ids = a['candidate_index'][ranking]
        # Freeze the ranking before using candidate outcomes for any budget result.
        atomic_npz(p, contract_sha256=np.asarray(contract),source_index=np.asarray(sid),
                   input_gradient=grad,move_score=move_score,ranked_candidate_index=candidate_ids,
                   surrogate_clean_scores=s,rival_class=np.asarray(rival),clean_class=np.asarray(ref['prediction']))
        _,tie_counts=np.unique(move_score,return_counts=True)
        first_change=np.flatnonzero(a['transition_code'][ranking]!=0)
        first_adverse=np.flatnonzero(a['transition_code'][ranking]==1)
        source_rows=[]
        for budget in budgets:
            k = min(budget,n); chosen=ranking[:k]
            row={'source_index':sid,'budget':int(budget),'selected_candidates':k,'candidates':n,
                 'initially_correct': bool(label==ref['prediction']),
                 'any_class_change_in_full_map':bool(np.any(a['transition_code']!=0)),
                 'any_adverse_in_full_map':bool(np.any(a['transition_code']==1)),
                 'found_class_change':bool(np.any(a['transition_code'][chosen]!=0)),
                 'found_adverse':bool(np.any(a['transition_code'][chosen]==1)),
                 'first_class_change_rank':int(first_change[0]+1) if len(first_change) else None,
                 'first_adverse_rank':int(first_adverse[0]+1) if len(first_adverse) else None,
                 'hypothetical_candidate_queries':k,
                 'selected_class_changes':int(np.sum(a['transition_code'][chosen]!=0)),
                 'selected_adverse':int(np.sum(a['transition_code'][chosen]==1))}
            source_rows.append(row)
        atomic_json(rowpath,{'contract_sha256':contract,'source_index':sid,'surrogate_forward_gate_pass':True,
                    'gate':gate,'gradient_nonzero_elements':int(np.count_nonzero(grad)),
                    'informative_gradient':bool(np.any(move_score!=0)),
                    'distinct_move_scores':len(tie_counts),'max_tie_fraction':float(tie_counts.max()/n),
                    'objective':'fixed-clean-runner-up score minus clean-predicted-class score',
                    'rival_class':rival,'backward_passes':1,'surrogate':'public ATan alpha=5, detach_reset=True',
                    'budgets':source_rows,'gradient_file_sha256':file_hash(p)})
        results.extend(source_rows);gate_rows.append(gate)
        print(f'[GRADIENT] source={sid} {len(gate_rows)}/100 grad_nonzero={np.count_nonzero(grad)} score_gate={err:.3g}',flush=True)
        if max_seconds and time.monotonic()-started>=max_seconds: break
    if results:
        with (root/'source_budget_results.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
        summary=[]
        for b in budgets:
            rows=[r for r in results if r['budget']==b]; adverse=[r for r in rows if r['any_adverse_in_full_map']]
            sensitive=[r for r in rows if r['any_class_change_in_full_map']]
            summary.append({'budget':b,'sources':len(rows),'adverse_sources':len(adverse),
                            'hypothetical_candidate_queries':sum(r['hypothetical_candidate_queries'] for r in rows),
                            'adverse_sources_detected':sum(r['found_adverse'] for r in adverse),
                            'class_sensitive_sources':len(sensitive),
                            'class_sensitive_sources_detected':sum(r['found_class_change'] for r in sensitive)})
        atomic_json(root/'summary.json',{'contract_sha256':contract,'completed':len(gate_rows)==100,
                    'sources':len(gate_rows),'single_clean_gradient_per_source':True,
                    'candidate_outcomes':'looked up without selection tuning in complete canonical EB18 map',
                    'new_candidate_forward_passes_this_reanalysis':0,
                    'clean_backward_passes_saved':len(gate_rows),'new_clean_backward_passes_this_invocation':new_backward,
                    'hypothetical_attack_cost':'One clean forward+backward per source plus min(k,N_source) candidate forward queries at budget k; reported outcomes reuse the canonical map retrospectively.',
                    'official_test_access':False,'budgets':summary,
                    'limitations':'One-step fixed-rival surrogate ranking; not an iterative or optimized retiming attack.'})


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=['gate','validate','gradient','all'],default='gate')
    ap.add_argument('--repo',required=True,type=Path);ap.add_argument('--checkpoint',required=True,type=Path)
    ap.add_argument('--validation',required=True,type=Path,help='ssc_valid.h5 or exact persistent/staged validation cache')
    ap.add_argument('--validation-manifest',type=Path,help='Required persistent source manifest when using staged arrays')
    ap.add_argument('--panel',required=True,type=Path,help='EB15R2 source folder or compact canonical panel NPZ')
    ap.add_argument('--geometry',required=True,type=Path,help='EB18_EXACT_B1_CANDIDATE_GEOMETRY.npz')
    ap.add_argument('--output',required=True,type=Path);ap.add_argument('--device',default='cpu')
    ap.add_argument('--threads',type=int,default=1);ap.add_argument('--score-atol',type=float,default=1e-4)
    ap.add_argument('--gate-limit',type=int,default=100,help='For --mode gate only: a partial diagnostic probe, never a full acceptance gate')
    ap.add_argument('--budgets',default='1,10,100,1000');ap.add_argument('--limit',type=int,default=9981)
    ap.add_argument('--validation-start',type=int,default=0);ap.add_argument('--validation-stop',type=int,default=9981)
    ap.add_argument('--max-seconds',type=float,default=0,help='Stop at a checkpoint boundary; 0 disables time budget')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.limit<=0 or args.threads<=0 or args.score_atol<0:ap.error('Invalid positive-count/tolerance argument')
    if not 0<=args.validation_start<args.validation_stop<=9981:ap.error('Validation range must lie within0..9981')
    if not 1<=args.gate_limit<=100 or (args.gate_limit!=100 and args.mode!='gate'):
        ap.error('A partial gate is only permitted in gate mode')
    budgets=sorted(set(map(int,args.budgets.split(','))))
    if not budgets or min(budgets)<=0:ap.error('Budgets must be positive')
    data=ValidationData(args.validation,args.validation_manifest)
    refs,panelhash=load_panel(args.panel)
    model,torch,functional,neuron,env=build_model(args.repo,args.checkpoint,args.device,args.threads)
    contract={'version':VERSION,'runner_sha256':file_hash(Path(__file__)),'checkpoint_sha256':CHECKPOINT_SHA,
              'state_dict_sha256':STATE_SHA,'upstream_commit':COMMIT,'source_blobs':PINNED,'data':data.provenance,
              'panel_clean_reference_sha256':panelhash,'geometry_sha256':file_hash(args.geometry),
              'environment':env,'score_atol':args.score_atol,'budgets':budgets,
              'preprocessing':'EB18 5ms/5units; source-specific horizon, first-event-relative time; no augmentation/padding to200',
              'inference':'unmodified public model semantic and computational batch size1, float32, deterministic, AMP/TF32off',
              'gradient':'one clean fixed-runner-up margin-drop surrogate backward; only LIF train mode; BN/dropout eval',
              'tie_break':'ascending canonical candidate_index','official_test_access':False}
    ch=json_hash(contract);cp=args.output/'contract.json'
    if cp.exists() and json.loads(cp.read_text()).get('contract_sha256')!=ch:
        raise ValueError('Output folder belongs to a different frozen contract; use a new folder')
    atomic_json(cp,{'contract_sha256':ch,'payload':contract})
    canonical_gate(model,torch,functional,data,refs,args.device,args.output,args.score_atol,ch,args.gate_limit)
    if args.mode in ('validate','all'):
        full_validation(model,torch,functional,data,args.device,args.output,ch,args.limit,args.max_seconds,args.validation_start,args.validation_stop)
    if args.mode in ('gradient','all'):
        gradient_ranking(model,torch,functional,neuron,data,refs,args.device,args.geometry,args.output,ch,
                         args.score_atol,budgets,min(args.limit,100),args.max_seconds)


if __name__=='__main__':
    main()
