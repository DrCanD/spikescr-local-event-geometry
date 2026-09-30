#!/usr/bin/env python3
"""34 clean target gradients/source, with canonical hard-forward parity.

Only saved validation inputs are read. Rankings are frozen before outcome lookup.
No model optimization, training, candidate forward, or test access is performed.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
DEFAULT_OLD=HERE.parent/'experiments/revision_work/experiments_20260929'
POLICIES=('max_directional_gain','max_predicted_target_margin')
BUDGETS=(1,10,100,1000)

def load_runner(root):
    path=root/'model_runner/run_singleton_gradient_v2.py'
    spec=importlib.util.spec_from_file_location('frozen_runner',path)
    r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
    return r

def geometry(root,refs):
    path=root.parent/'data/eb18/EB18_EXACT_B1_CANDIDATE_GEOMETRY.npz'
    with np.load(path,allow_pickle=False) as z:
        g={k:z[k] for k in ('source_index','candidate_index','from_bin','to_bin','feature',
                           'multiplicity','candidate_prediction','clean_prediction','label','transition_code')}
    if len(g['source_index'])!=725070 or set(map(int,np.unique(g['source_index'])))!=set(refs):
        raise ValueError('Frozen candidate map/panel mismatch')
    return path,g

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=DEFAULT_OLD)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--device',default='cpu');ap.add_argument('--threads',type=int,default=1)
    ap.add_argument('--prepare',action='store_true',help='Write contract and complete 100-source gate, then stop.')
    ap.add_argument('--source-start',type=int,default=0);ap.add_argument('--source-stop',type=int,default=100)
    args=ap.parse_args()
    if not 0<=args.source_start<args.source_stop<=100:ap.error('Ordinal source range must lie in [0,100)')
    root=args.root.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    r=load_runner(root);data=r.ValidationData(root/'assets/valid_cache')
    refs,panelhash=r.load_panel(root/'assets/canonical_clean_panel.npz')
    gp,g=geometry(root,refs)
    model,torch,functional,neuron,env=r.build_model(root/'assets/SpikeSCR',root/'assets/best_valid.pt',args.device,args.threads)
    payload=dict(version='multi-target-20260930-v1',script_sha256=r.file_hash(__file__),
        frozen_runner_sha256=r.file_hash(root/'model_runner/run_singleton_gradient_v2.py'),
        checkpoint_sha256=r.CHECKPOINT_SHA,state_dict_sha256=r.STATE_SHA,source_blobs=r.PINNED,
        data=data.provenance,panel_sha256=panelhash,geometry_sha256=r.file_hash(gp),environment=env,
        targets='All 34 classes other than frozen clean predicted class; increasing class index.',
        objective='For each c != clean_prediction: sum_time softmax(logits)[c] - sum_time softmax(logits)[clean_prediction].',
        direction='input_gradient[to,feature] - input_gradient[from,feature]',
        policies={'max_directional_gain':'max over all 34 directional gains',
                  'max_predicted_target_margin':'max over all 34 (clean target margin + directional gain)'},
        tie_break='ascending canonical candidate_index',surrogate='pinned public ATan alpha=5; detach_reset=True',
        acceptance='100-source canonical score tolerance 1e-4; every surrogate forward bit-exact to its current eval forward, including all LIF spikes.',
        candidate_outcomes='Frozen and saved ranks are looked up retrospectively in the canonical map.',
        budgets_per_source=list(BUDGETS),official_test_access=False)
    ch=r.json_hash(payload);cp=out/'contract.json'
    if cp.exists() and json.loads(cp.read_text())['contract_sha256']!=ch:
        raise ValueError('Output contract differs; use a new directory')
    if args.prepare:
        r.atomic_json(cp,dict(contract_sha256=ch,payload=payload))
        r.canonical_gate(model,torch,functional,data,refs,args.device,out,1e-4,ch)
        print('[PREPARED]',ch,flush=True);return
    if not cp.exists(): raise ValueError('Run --prepare before workers')
    with np.load(out/'canonical_panel_gate.npz',allow_pickle=False) as z:
        if not bool(z['pass']) or str(z['contract_sha256'])!=ch or len(z['source_index'])!=100:
            raise ValueError('Missing complete contract-bound parity gate')
        runtime={int(sid):z['scores'][j] for j,sid in enumerate(z['source_index'])}
        lif_names=list(map(str,z['lif_names']))
        spikes={int(sid):list(map(str,z['lif_spike_sha256'][j])) for j,sid in enumerate(z['source_index'])}
    results=out/'sources';results.mkdir(exist_ok=True)
    for sid in sorted(refs)[args.source_start:args.source_stop]:
        p=results/f'source_{sid:05d}.npz';jp=p.with_suffix('.json')
        if p.exists() and jp.exists():
            row=json.loads(jp.read_text())
            if row.get('contract_sha256')!=ch or row.get('ranking_file_sha256')!=r.file_hash(p):
                raise ValueError('Saved source contract/ranking digest differs')
            print('[REUSED]',sid,flush=True);continue
        started=time.monotonic();x,label=data.get(sid);ref=refs[sid]
        a={k:v[g['source_index']==sid] for k,v in g.items()};o=np.argsort(a['candidate_index'],kind='stable')
        a={k:v[o] for k,v in a.items()};n=len(o)
        frm,to,feat=(a[k].astype(np.int64) for k in ('from_bin','to_bin','feature'))
        expected_n=2*np.count_nonzero(x)-np.count_nonzero(x[0])-np.count_nonzero(x[-1])
        if n!=expected_n or not np.array_equal(a['candidate_index'],np.arange(n)):
            raise ValueError('Candidate coverage/index mismatch')
        if np.any(np.abs(to-frm)!=1) or np.any(frm<0) or np.any(to<0) or np.any(frm>=len(x)) or np.any(to>=len(x)):
            raise ValueError('Illegal move time coordinates')
        if np.any(feat<0) or np.any(feat>=140) or np.any(x[frm,feat]<=0) or not np.array_equal(x[frm,feat],a['multiplicity']):
            raise ValueError('Illegal move feature or multiplicity')
        if np.any(a['label']!=label) or np.any(a['clean_prediction']!=ref['prediction']):
            raise ValueError('Map clean labels differ')
        model.eval();functional.reset_net(model)
        for m in model.modules():
            if isinstance(m,neuron.LIFNode):m.train(True)
        if any(m.training for m in model.modules() if isinstance(m,(torch.nn.modules.batchnorm._BatchNorm,torch.nn.Dropout))):
            raise ValueError('BN/dropout must remain eval')
        inp=torch.from_numpy(x[None]).to(args.device).requires_grad_(True)
        mask=torch.ones((1,len(x)),dtype=torch.bool,device=args.device)
        hashes,handles=r.install_spike_hash_hooks(model)
        try:
            with torch.enable_grad():
                scores=torch.softmax(model(inp,mask).float(),dim=2).sum(dim=0)[0]
                s=scores.detach().cpu().numpy().astype(np.float32)
                if not np.array_equal(s,runtime[sid]) or sorted(hashes)!=lif_names or [hashes[k] for k in lif_names]!=spikes[sid]:
                    raise RuntimeError(f'Hard-forward parity failed for {sid}; no accepted gradient')
                if int(s.argmax())!=ref['prediction'] or np.max(np.abs(s-ref['scores']))>1e-4:
                    raise RuntimeError('Canonical score/class gate failed')
                clean=ref['prediction'];targets=np.array([c for c in range(35) if c!=clean],dtype=np.int16)
                all_gains=[];input_gradients=[]
                for j,target in enumerate(targets):
                    grad=torch.autograd.grad(scores[int(target)]-scores[clean],inp,retain_graph=j<33)[0][0]
                    grad=grad.detach().cpu().numpy().astype(np.float32)
                    if not np.isfinite(grad).all():raise ValueError('Nonfinite gradient')
                    input_gradients.append(grad);all_gains.append(grad[to,feat]-grad[frm,feat])
        finally:
            for h in handles:h.remove()
            model.eval();functional.reset_net(model)
        gains=np.stack(all_gains);margins=s[targets]-s[clean]
        directional=gains.max(axis=0)
        predicted=(gains+margins[:,None]).max(axis=0)
        ranks=np.stack([np.lexsort((a['candidate_index'],-v)) for v in (directional,predicted)])
        # No candidate outcomes are read to choose or alter these ranks.
        r.atomic_npz(p,contract_sha256=np.asarray(ch),source_index=np.asarray(sid),
            targets=targets,clean_class=np.asarray(clean),clean_scores=s,input_gradients=np.stack(input_gradients),
            candidate_index=a['candidate_index'],target_directional_gains=gains,
            policy_names=np.asarray(POLICIES),policy_move_scores=np.stack([directional,predicted]),
            ranked_candidate_index=a['candidate_index'][ranks])
        row=dict(contract_sha256=ch,source_index=sid,label=label,clean_prediction=clean,
            backward_passes=34,surrogate_hard_forward_bit_parity=True,
            canonical_max_abs_error=float(np.max(np.abs(s-ref['scores']))),
            input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),ranking_file_sha256=r.file_hash(p),
            elapsed_seconds=time.monotonic()-started,targets=targets.tolist())
        r.atomic_json(jp,row)
        print(f'[MULTI] source={sid} backwards=34 elapsed={row["elapsed_seconds"]:.2f}s',flush=True)

if __name__=='__main__':main()
