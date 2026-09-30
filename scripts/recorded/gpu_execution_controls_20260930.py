#!/usr/bin/env python3
"""Same-runtime GPU execution controls for frozen SSC validation, no test.

Full conditions: native B1, padding-matched B1, B256 and reversed-order B256.
CPU smoke mode is limited to four sources and cannot produce scientific results.
Cross-device disagreement is measured, never used as a reason to change weights.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
DEFAULT_ROOT=HERE.parent/'experiments/revision_work/experiments_20260929'
CONDITIONS=('B1_native','B1_padding_matched','B256','B256_reversed')

def load_runner(root):
    spec=importlib.util.spec_from_file_location('frozen_runner',root/'model_runner/run_singleton_gradient_v2.py')
    r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r);return r

def paired(y,a,b):
    different=a!=b;ac=a==y;bc=b==y
    return dict(predicted_label_disagreements=int(different.sum()),a_correct_b_wrong=int((ac&~bc).sum()),
        a_wrong_b_correct=int((~ac&bc).sum()),both_wrong_different_label=int((~ac&~bc&different).sum()),
        both_correct=int((ac&bc).sum()),both_wrong=int((~ac&~bc).sum()))

def metrics(y,p):
    cm=np.zeros((35,35),dtype=int);np.add.at(cm,(y,p),1)
    recall=np.divide(cm.diagonal(),cm.sum(axis=1),out=np.zeros(35,dtype=float),where=cm.sum(axis=1)>0)
    f1=np.divide(2*cm.diagonal(),cm.sum(axis=0)+cm.sum(axis=1),out=np.zeros(35,dtype=float),where=cm.sum(axis=0)+cm.sum(axis=1)>0)
    return dict(correct=int((y==p).sum()),samples=len(y),accuracy=float((y==p).mean()),
        balanced_accuracy=float(recall.mean()),macro_f1=float(f1.mean()))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=DEFAULT_ROOT);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--cpu-smoke',action='store_true');ap.add_argument('--limit',type=int,default=9981)
    a=ap.parse_args()
    if not 1<=a.limit<=9981:ap.error('Limit must be 1..9981')
    if a.cpu_smoke and a.limit>4:ap.error('CPU smoke cannot exceed four sources')
    root=a.root.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    r=load_runner(root);data=r.ValidationData(root/'assets/valid_cache')
    refs,panelhash=r.load_panel(root/'assets/canonical_clean_panel.npz')
    import torch
    if not a.cpu_smoke and not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required. Select a GPU runtime; no CPU fallback is allowed.')
    device='cpu' if a.cpu_smoke else 'cuda:0'
    model,torch,functional,neuron,env=r.build_model(root/'assets/SpikeSCR',root/'assets/best_valid.pt',device,1)
    if not a.cpu_smoke:
        props=torch.cuda.get_device_properties(0)
        env['gpu_properties']=dict(name=props.name,total_memory_bytes=props.total_memory,
            compute_capability=[props.major,props.minor],uuid=str(getattr(props,'uuid','unavailable')))
    cpu_path=root/'results_cpu_v2/validation_merged/validation_B1_predictions.npz'
    batch_path=root/'assets/benchmark_reference/validation_predictions.npz'
    payload=dict(version='same-runtime-execution-controls-20260930-v1',
        script_sha256=r.file_hash(__file__),runner_sha256=r.file_hash(root/'model_runner/run_singleton_gradient_v2.py'),
        checkpoint_sha256=r.CHECKPOINT_SHA,state_dict_sha256=r.STATE_SHA,source_blobs=r.PINNED,
        data=data.provenance,environment=env,panel_clean_reference_sha256=panelhash,
        cpu_B1_reference_sha256=r.file_hash(cpu_path),archived_GPU_B256_reference_sha256=r.file_hash(batch_path),
        split='validation only',official_test_access=False,source_count_requested=a.limit,cpu_smoke=a.cpu_smoke,
        batch_groups='ascending source IDs, contiguous groups of 256; last group retains its native size',
        scores='Pinned public readout: sum over every padded time bin of softmax(logits); no new masking of output scores.',
        conditions={'B1_native':'one source, its native input horizon',
            'B1_padding_matched':'one source, padded with zero counts to its corresponding B256 group horizon; same valid-input mask',
            'B256':'all sources in the same contiguous group, zero padding to group maximum horizon',
            'B256_reversed':'same padded tensors and same B/T shapes, with source order reversed and restored after prediction'},
        execution='identical frozen model, GPU, runtime, float32, eval, state reset per forward; AMP and TF32 disabled; deterministic algorithms',
        role='Estimate hardware/runtime difference at B1, padding difference at B1, and batched execution difference at fixed padding. Reversed B256 keeps group composition and tensor shape fixed.',
        interpretation='Historical vs fresh B256 drift is reported separately. Fixed GPU hardware does not by itself exclude all batch-shape numerical effects.')
    ch=r.json_hash(payload);cp=out/'contract.json'
    if cp.exists() and json.loads(cp.read_text())['contract_sha256']!=ch:
        raise ValueError('Output belongs to a different runtime/contract; choose a new directory')
    r.atomic_json(cp,dict(contract_sha256=ch,payload=payload))
    parts=[];started=time.monotonic();groups=out/'groups';groups.mkdir(exist_ok=True)

    def batch_scores(inputs,lengths):
        functional.reset_net(model)
        try:
            inp=torch.from_numpy(inputs).to(device)
            mask=torch.arange(inputs.shape[1],device=device)[None,:] < torch.as_tensor(np.ascontiguousarray(lengths),device=device)[:,None]
            with torch.no_grad():
                score=torch.softmax(model(inp,mask).float(),dim=2).sum(dim=0).cpu().numpy().astype(np.float32)
            if not np.isfinite(score).all() or score.shape!=(len(inputs),35):
                raise ValueError('Invalid scores')
            return score
        finally:functional.reset_net(model)

    for lo in range(0,a.limit,256):
        hi=min(lo+256,a.limit);p=groups/f'group_{lo:05d}_{hi:05d}.npz'
        if p.exists():
            with np.load(p,allow_pickle=False) as z:
                if str(z['contract_sha256'])!=ch or not np.array_equal(z['source_index'],np.arange(lo,hi)):
                    raise ValueError('Saved chunk provenance mismatch')
                rows={k:z[k] for k in z.files if k!='contract_sha256'}
        else:
            loaded=[data.get(sid) for sid in range(lo,hi)]
            lengths=np.array([len(x) for x,y in loaded],dtype=np.int32);H=int(lengths.max())
            inputs=np.zeros((hi-lo,H,140),dtype=np.float32)
            for j,(x,y) in enumerate(loaded):inputs[j,:len(x)]=x
            s256=batch_scores(inputs,lengths)
            srev=batch_scores(np.array(inputs[::-1],copy=True),lengths[::-1])[::-1]
            native=[];padded=[];repeat_ids=[]
            for j,(x,y) in enumerate(loaded):
                native.append(batch_scores(x[None],np.array([len(x)]))[0])
                padded.append(batch_scores(inputs[j:j+1],lengths[j:j+1])[0])
                # Ten identical-input singleton replays establish within-runtime repeatability.
                if lo+j in sorted(refs)[:10]:
                    replay=batch_scores(x[None],np.array([len(x)]))[0]
                    if not np.array_equal(replay,native[-1]):raise RuntimeError('Identical GPU singleton replay changed scores')
                    repeat_ids.append(lo+j)
            rows=dict(source_index=np.arange(lo,hi,dtype=np.int32),
                label=np.array([y for x,y in loaded],dtype=np.int16),native_horizon=lengths,
                padded_horizon=np.full(hi-lo,H,dtype=np.int32),
                input_sha256=np.array([hashlib.sha256(x.tobytes()).hexdigest() for x,y in loaded]),
                B1_native_scores=np.stack(native),B1_padding_matched_scores=np.stack(padded),
                B256_scores=s256,B256_reversed_scores=srev,replayed_source_index=np.array(repeat_ids,dtype=int))
            r.atomic_npz(p,contract_sha256=np.asarray(ch),**rows)
        parts.append(rows)
        r.atomic_json(out/'progress.json',dict(contract_sha256=ch,completed_sources=hi,
            requested_sources=a.limit,scientific_complete=hi==9981 and not a.cpu_smoke,
            elapsed_seconds=time.monotonic()-started,official_test_access=False))
        print(f'[EXECUTION CONTROL] {hi}/{a.limit} elapsed={time.monotonic()-started:.1f}s',flush=True)
    combined={k:np.concatenate([v[k] for v in parts]) for k in parts[0]}
    ids=combined['source_index'];y=combined['label'];assert np.array_equal(ids,np.arange(a.limit))
    with np.load(cpu_path,allow_pickle=False) as z:
        assert np.array_equal(z['source_index'],np.arange(9981)) and np.array_equal(z['label'][:a.limit],y)
        assert np.array_equal(z['horizon'][:a.limit],combined['native_horizon'])
        assert np.array_equal(z['input_sha256'][:a.limit],combined['input_sha256'])
        cpu=z['prediction'][:a.limit];cpu_scores=z['scores'][:a.limit]
    with np.load(batch_path,allow_pickle=False) as z:
        # Recognize exact archived layouts; fail on any unrecognized format.
        keys=set(z.files)
        if {'labels','predictions','source_indices'}<=keys:
            assert np.array_equal(z['source_indices'],np.arange(9981))
            by=z['labels'];bp=z['predictions']
        elif {'y_true','y_pred'}<=keys:by=z['y_true'];bp=z['y_pred']
        elif {'label','prediction'}<=keys:by=z['label'];bp=z['prediction']
        else:raise ValueError('Unknown archived benchmark prediction schema')
        assert len(by)==9981 and np.array_equal(by[:a.limit],y)
        bp=bp[:a.limit]
    preds={c:combined[c+'_scores'].argmax(axis=1) for c in CONDITIONS}
    comparisons=[('CPU_B1','B1_native',cpu,preds['B1_native']),
        ('B1_native','B1_padding_matched',preds['B1_native'],preds['B1_padding_matched']),
        ('B1_padding_matched','B256',preds['B1_padding_matched'],preds['B256']),
        ('B1_native','B256',preds['B1_native'],preds['B256']),
        ('B256','B256_reversed',preds['B256'],preds['B256_reversed']),
        ('archived_GPU_B256','B256',bp,preds['B256'])]
    pairs=[dict(a=an,b=bn,**paired(y,pa,pb)) for an,bn,pa,pb in comparisons]
    panel_ids=np.array([sid for sid in sorted(refs) if sid<a.limit],dtype=int)
    panel_scores=combined['B1_native_scores'][panel_ids]
    panel_ref=np.stack([refs[int(sid)]['scores'] for sid in panel_ids]) if len(panel_ids) else np.empty((0,35))
    summary=dict(contract_sha256=ch,completed=a.limit==9981 and not a.cpu_smoke,
        cpu_smoke=a.cpu_smoke,samples=len(ids),environment=env,conditions={c:metrics(y,preds[c]) for c in CONDITIONS},
        CPU_B1_reference=metrics(y,cpu),archived_GPU_B256_reference=metrics(y,bp),paired_comparisons=pairs,
        canonical_panel=dict(sources=len(panel_ids),predicted_label_disagreements=int(np.sum(panel_scores.argmax(axis=1)!=panel_ref.argmax(axis=1))) if len(panel_ids) else 0,
            max_abs_score_difference=float(np.max(np.abs(panel_scores-panel_ref))) if len(panel_ids) else None),
        GPU_B1_vs_CPU_B1_score_max_abs=float(np.max(np.abs(combined['B1_native_scores']-cpu_scores))),
        identical_singleton_replays=len(combined['replayed_source_index']),official_test_access=False,
        interpretation=payload['interpretation'],elapsed_seconds_this_invocation=time.monotonic()-started)
    for c in CONDITIONS:combined[c+'_prediction']=preds[c]
    combined['CPU_B1_prediction']=cpu;combined['archived_GPU_B256_prediction']=bp
    r.atomic_npz(out/'predictions.npz',contract_sha256=np.asarray(ch),**combined)
    pd.DataFrame(pairs).to_csv(out/'paired_comparisons.csv',index=False)
    r.atomic_json(out/'summary.json',summary)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
