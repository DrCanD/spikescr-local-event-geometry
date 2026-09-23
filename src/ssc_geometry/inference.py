"""Explicit singleton replay of the frozen model; no model selection or training."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from .analysis import load_and_validate, require
from .core import move_one_count
from .io import read_json, write_json, load_npz, sha256_file


def execution_digest(root: Path) -> str:
    h=hashlib.sha256()
    for directory in ('src','configs','scripts'):
        for p in sorted((root/directory).rglob('*')):
            if (p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'
                    and not any(part.endswith('.egg-info') for part in p.parts)):
                h.update(p.relative_to(root).as_posix().encode());h.update(p.read_bytes())
    for p in sorted([root/'pyproject.toml', *root.glob('requirements-*.txt')]):
        h.update(p.relative_to(root).as_posix().encode());h.update(p.read_bytes())
    return h.hexdigest()


def compare_point(scores, traces, patch_scores, source, global_index, b, patch_lookup):
    """Compare observed outputs, never substitute reference outputs for computation."""
    g=b['geometry']; ref=b['internal'];cfg=b['config'];i=global_index;errors=[]
    tol=cfg['source_score_atol']; ttol=cfg['replay_trace_atol']
    if not np.isfinite(scores).all() or scores.shape!=(35,):raise ValueError('Invalid observed score vector')
    if traces.shape!=(7,6) or not np.isfinite(traces).all():raise ValueError('Invalid observed trace metrics')
    pred=int(scores.argmax());label=int(g['label'][i]);clean_pred=int(g['clean_prediction'][i])
    if pred!=int(g['candidate_prediction'][i]):errors.append('candidate_prediction')
    differences={}
    for name,target in [('candidate_clean_class_margin',clean_pred),('candidate_true_class_margin',label)]:
        other=scores.copy();other[target]=-np.inf;value=float(scores[target]-other.max())
        diff=abs(value-float(g[name][i]));differences[name]=diff
        if diff>tol:errors.append(name)
    top=np.sort(scores);diff=abs(float(top[-1]-top[-2])-float(g['candidate_top1_margin'][i]));differences['candidate_top1_margin']=diff
    if diff>tol:errors.append('candidate_top1_margin')
    clean_scores=b['clean_scores'][source]
    delta=scores-clean_scores
    denominator=np.linalg.norm(scores)*float(np.linalg.norm(clean_scores))
    cosine=float((scores @ clean_scores)/denominator) if denominator>0 else 1.0
    scalar={'delta_clean_class_score':float(delta[clean_pred]),'delta_true_class_score':float(delta[label]),
            'score_l2_change':float(np.linalg.norm(delta)), 'score_linf_change':float(np.max(np.abs(delta))),
            'score_cosine_distance':float(np.float32(1.0-cosine))}
    for name,value in scalar.items():
        diff=abs(value-float(g[name][i]));differences[name]=diff
        if diff>tol:errors.append(name)
    td=np.abs(traces-ref['trace_metrics'][i]);differences['max_trace_absolute_difference']=float(td.max())
    if not np.isfinite(td).all() or float(td.max())>ttol:errors.append('trace_metrics')
    pi=patch_lookup.get((source,int(g['candidate_index'][i])))
    if pi is not None and patch_scores is None:
        errors.append('missing_patch_scores')
    if patch_scores is not None:
        if not np.isfinite(patch_scores).all() or patch_scores.shape!=(7,35):raise ValueError('Invalid patch scores')
        if pi is not None:
            pd=np.abs(patch_scores-ref['patch_scores'][pi]);differences['max_patch_score_absolute_difference']=float(pd.max())
            if float(pd.max())>tol:errors.append('patch_scores')
            if not np.array_equal(patch_scores.argmax(axis=1),ref['patch_prediction'][pi]):errors.append('patch_predictions')
        control=np.abs(patch_scores[[0,3,6]]-b['clean_scores'][source])
        differences['positive_control_max_absolute_difference']=float(control.max())
        if float(control.max())>tol:errors.append('positive_controls')
    return {'status':'PASS' if not errors else 'FAIL','errors':errors,**differences}


def validate_cached_source(cached, summary, source, candidate_ids, patch_ids, clean_reference, tolerance):
    """Require complete stored outputs before any source is counted as resumed."""
    if int(cached['source_index']) != source or summary.get('source_index') != source:
        raise ValueError('Cached source identity mismatch')
    if cached['candidate_index'].tolist() != candidate_ids:
        raise ValueError('Resume candidate coverage mismatch')
    if cached['patch_candidate_index'].tolist() != patch_ids:
        raise ValueError('Resume replacement coverage mismatch')
    if summary.get('candidates') != len(candidate_ids) or summary.get('patched_candidates') != len(patch_ids):
        raise ValueError('Resume summary coverage mismatch')
    expected_shapes = {
        'scores': (len(candidate_ids),35),
        'trace_metrics': (len(candidate_ids),7,6),
        'patch_scores': (len(patch_ids),7,35),
        'clean_scores': (35,),
    }
    for name, shape in expected_shapes.items():
        if cached[name].shape != shape or not np.isfinite(cached[name]).all():
            raise ValueError('Invalid cached output: '+name)
    if (float(np.abs(cached['clean_scores']-clean_reference).max()) > tolerance
            or int(cached['clean_scores'].argmax()) != int(clean_reference.argmax())):
        raise ValueError('Cached clean score parity failed')
    return {int(ci):value for ci,value in zip(cached['patch_candidate_index'],cached['patch_scores'])}


def run_audit(root: Path, source_path: Path, out: Path, mode: str='preflight',
              device_name: str='cuda', allow_nonreference: bool=False,
              sources: list[int] | None=None, max_candidates: int | None=None,
              resume: bool=False) -> dict:
    from .upstream import load_model
    from . import _forward_kernel as fk
    import torch
    b=load_and_validate(root);g=b['geometry'];cfg=b['config']
    all_ids=[row['source_index'] for row in b['manifest']]
    if sources is not None and (len(sources)!=len(set(sources)) or any(s not in all_ids for s in sources)):
        raise ValueError('Sources must be unique IDs from the fixed panel')
    if max_candidates is not None and max_candidates<1:raise ValueError('Candidate limit must be positive')
    header={'schema_version':1,'mode':mode,'sources':sources,'max_candidates':max_candidates,
        'device':device_name,'allow_nonreference_environment':allow_nonreference,
        'execution_digest':execution_digest(root),'integrity_manifest_sha256':sha256_file(root/'checksums/manifest.json'),
        'score_atol':cfg['source_score_atol'],'trace_atol':cfg['replay_trace_atol']}
    manifest_path=out/'run_manifest.json'
    if resume:
        if not manifest_path.is_file() or read_json(manifest_path)!=header:raise ValueError('Resume contract mismatch')
    else:write_json(manifest_path,header)
    if resume and (out/'FAILED.json').exists():
        k=1
        while (out/f'prior_failure_{k:03d}.json').exists():k+=1
        (out/'FAILED.json').replace(out/f'prior_failure_{k:03d}.json')
    model,device,stages,environment=load_model(root,source_path,device_name,allow_nonreference)
    if resume and (out/'environment.json').is_file() and read_json(out/'environment.json')!=environment:
        raise ValueError('Resume environment mismatch; use a new output directory')
    write_json(out/'environment.json',environment)
    lookup={(int(s),int(c)):i for i,(s,c) in enumerate(zip(b['internal']['patch_source_index'],b['internal']['patch_candidate_index']))}
    # Fixed probes cover all four transition types in the recorded preflight.
    probes={7043:[0],4284:[1920],8189:[0],8998:[0]}
    selected_ids=[s for s in all_ids if (sources is None or s in sources) and (mode!='preflight' or s in probes)]
    if not selected_ids:raise ValueError('No preflight probe belongs to the requested sources')
    summaries=[];new_forwards=0;new_candidates=0
    for sid in selected_ids:
        indices=np.flatnonzero(g['source_index']==sid)
        candidate_ids=probes[sid] if mode=='preflight' else list(range(len(indices)))
        if max_candidates is not None:candidate_ids=candidate_ids[:max_candidates]
        archive_path=out/f'source_{sid:05d}.npz';summary_path=out/f'source_{sid:05d}.json'
        if resume and summary_path.is_file():
            previous=read_json(summary_path)
            if previous.get('status')!='PASS' or previous.get('execution_digest')!=header['execution_digest']:
                raise ValueError('Invalid resume source summary')
            if not archive_path.is_file() or sha256_file(archive_path)!=previous.get('archive_sha256'):
                raise ValueError('Resume archive checksum mismatch')
            cached=load_npz(archive_path)
            patch_ids=[ci for ci in candidate_ids if mode=='preflight' or g['transition_code'][indices[ci]]!=0]
            cached_patch=validate_cached_source(cached,previous,sid,candidate_ids,patch_ids,b['clean_scores'][sid],cfg['source_score_atol'])
            for j,ci in enumerate(candidate_ids):
                check=compare_point(cached['scores'][j],cached['trace_metrics'][j],cached_patch.get(ci),sid,int(indices[ci]),b,lookup)
                if check['status']!='PASS':raise ValueError('Cached outputs fail canonical comparison')
            summaries.append(previous);continue
        x=b['inputs'][f'source_{sid:05d}'];batch=torch.from_numpy(x.astype(np.float32)[None])
        clean_scores,clean_traces=fk.captured_model_forward(model,batch,device,stages);new_forwards+=1
        clean_diff=float(np.abs(clean_scores[0]-b['clean_scores'][sid]).max())
        if clean_diff>cfg['source_score_atol'] or int(clean_scores.argmax())!=int(b['clean_scores'][sid].argmax()):
            write_json(out/'FAILED.json',{'status':'FAIL','source_index':sid,'reason':'clean_score_parity',
                'max_absolute_difference':clean_diff,'score_atol':cfg['source_score_atol'],
                'observed_prediction':int(clean_scores[0].argmax()),
                'reference_prediction':int(b['clean_scores'][sid].argmax())})
            raise ValueError('Clean score parity failed; candidate evaluation stopped')
        axes=fk.infer_trace_batch_axes(clean_traces,1)
        if mode=='preflight':
            for stage in stages:
                identity=fk.patched_model_scores(model,batch,device,stage,clean_traces[stage['stage']],axes[stage['stage']])[0];new_forwards+=1
                if float(np.abs(identity-clean_scores[0]).max())>cfg['source_score_atol']:
                    raise ValueError('Clean identity patch failed')
        observed_scores=[];observed_traces=[];patched_indices=[];observed_patches=[];maxima={}
        for ci in candidate_ids:
            gi=int(indices[ci]);z=move_one_count(x,int(g['from_bin'][gi]),int(g['to_bin'][gi]),int(g['feature'][gi]))
            candidate=torch.from_numpy(z.astype(np.float32)[None])
            score,trace=fk.captured_model_forward(model,candidate,device,stages);new_forwards+=1;new_candidates+=1
            metrics=np.stack([fk.singleton_trace_metrics(trace[stage['stage']],clean_traces[stage['stage']]) for stage in stages])
            patches=None
            if g['transition_code'][gi]!=0 or mode=='preflight':
                patches=np.stack([fk.patched_model_scores(model,candidate,device,stage,clean_traces[stage['stage']],axes[stage['stage']])[0] for stage in stages]);new_forwards+=7
            comparison=compare_point(score[0],metrics,patches,sid,gi,b,lookup)
            if comparison['status']!='PASS':
                write_json(out/'FAILED.json',{'source_index':sid,'candidate_index':ci,**comparison})
                raise ValueError('Candidate conformance failed; observed outputs were not substituted by references')
            for key,value in comparison.items():
                if isinstance(value,(float,int)):maxima[key]=max(maxima.get(key,0.),value)
            observed_scores.append(score[0]);observed_traces.append(metrics)
            if patches is not None:patched_indices.append(ci);observed_patches.append(patches)
        temp=archive_path.with_suffix('.tmp.npz')
        np.savez_compressed(temp,source_index=np.asarray(sid),candidate_index=np.array(candidate_ids,dtype=np.int32),
            clean_scores=clean_scores[0],scores=np.array(observed_scores,dtype=np.float32),trace_metrics=np.array(observed_traces,dtype=np.float32),
            patch_candidate_index=np.array(patched_indices,dtype=np.int32),patch_scores=np.array(observed_patches,dtype=np.float32).reshape(-1,7,35))
        temp.replace(archive_path)
        summary={'status':'PASS','source_index':sid,'candidates':len(candidate_ids),'patched_candidates':len(patched_indices),
            'archive_sha256':sha256_file(archive_path),'execution_digest':header['execution_digest'],'clean_score_max_absolute_difference':clean_diff,'maxima':maxima}
        write_json(summary_path,summary);summaries.append(summary)
        print(f"Source {sid}: {len(candidate_ids)} candidates compared",flush=True)
    total=sum(s['candidates'] for s in summaries);patched=sum(s['patched_candidates'] for s in summaries)
    full=mode=='full' and total==725070 and len(summaries)==100 and patched==25820
    report={'status':'PASS','scope':mode,'complete_canonical_panel':full,'sources':len(summaries),'candidates':total,
        'patched_candidates':patched,'new_candidates_this_invocation':new_candidates,'new_forward_passes_this_invocation':new_forwards,
        'raw_official_test_opened':False,'training_performed':False,'environment':environment,
        'execution_digest':header['execution_digest'],'integrity_manifest_sha256':header['integrity_manifest_sha256'],
        'interpretation':'Conformance of a frozen checkpoint, not an independent training replication or universal cross-hardware guarantee.'}
    write_json(out/'conformance_report.json',report)
    return report
