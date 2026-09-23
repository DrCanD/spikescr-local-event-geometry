"""Regenerate statistics from released arrays, independently of reference summaries."""
from __future__ import annotations
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any
import numpy as np
from .core import classify_transitions, enumerate_moves, move_one_count, validate_input
from .io import load_npz, read_numeric_csv, read_json, write_json, write_csv, compare_tree, compare_internal_statistics
from ._source_geometry import derive_source_geometry
from ._geometry_statistics import geometry_aggregates
from ._internal_statistics import recompute_internal
from ._kernel import spearman_rho

GROUPS = ['class_preserved', 'adverse', 'corrective', 'lateral']


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def benchmark_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    require(labels.ndim == 1 and labels.shape == predictions.shape, 'Benchmark shape mismatch')
    require(all(a.dtype.kind in 'iu' and np.all((a >= 0) & (a < 35)) for a in (labels, predictions)), 'Invalid class IDs')
    confusion = np.zeros((35, 35), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    support, predicted = confusion.sum(axis=1), confusion.sum(axis=0)
    diagonal = np.diag(confusion)
    recall = np.divide(diagonal, support, out=np.zeros(35), where=support > 0)
    precision = np.divide(diagonal, predicted, out=np.zeros(35), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(35), where=(precision + recall) > 0)
    present = support > 0
    return dict(samples=len(labels), correct=int((labels == predictions).sum()), accuracy=float((labels == predictions).mean()),
                balanced_accuracy=float(recall[present].mean()), macro_f1=float(f1[present].mean()),
                worst_class_recall=float(recall[present].min()), class_support=support.tolist(), class_recall=recall.tolist(),
                class_precision=precision.tolist(), class_f1=f1.tolist(), confusion=confusion.tolist())


def load_and_validate(root: Path) -> dict[str, Any]:
    """Validate correspondence of every record, panel source, and patched output."""
    config = read_json(root/'configs/experiment.json')
    geometry = load_npz(root/'data/neighborhood/candidate_records.npz')
    internal = load_npz(root/'data/internal/activation_metrics_and_replacements.npz')
    inputs = load_npz(root/'data/panel/inputs.npz')
    clean_file = load_npz(root/'data/panel/clean_scores.npz')
    manifest = read_numeric_csv(root/'data/panel/manifest.csv')
    stored_source_rows = read_numeric_csv(root/'data/neighborhood/source_geometry.csv')
    n = len(geometry['source_index'])
    require(n == 725070, 'Unexpected candidate count')
    require(len(manifest) == 100 and len(stored_source_rows) == 100, 'Expected 100 panel records')
    require(len({row['source_index'] for row in manifest}) == 100, 'Duplicate source')
    quotas=Counter(row['label'] for row in manifest)
    require(len(quotas)==35 and set(quotas.values()).issubset({2,3}), 'Fixed label quota mismatch')
    require(set(inputs) == {f"source_{row['source_index']:05d}" for row in manifest}, 'Panel key mismatch')
    require(clean_file['scores'].shape == (100,35), 'Clean score dimensions')
    require(np.isfinite(clean_file['scores']).all(), 'Nonfinite clean scores')
    require(clean_file['source_index'].tolist() == [row['source_index'] for row in manifest], 'Clean source order mismatch')
    clean = {int(s): v for s,v in zip(clean_file['source_index'], clean_file['scores'])}
    for key, arr in geometry.items():
        require(arr.shape == (n,), f'Invalid geometry field shape {key}')
        require(arr.dtype.kind in 'iuf' and np.isfinite(arr).all(), f'Invalid geometry numeric field {key}')
    codes = classify_transitions(geometry['label'], geometry['clean_prediction'], geometry['candidate_prediction'])
    require(np.array_equal(codes, geometry['transition_code']), 'Transition partition mismatch')
    for key in ('source_index','candidate_index','transition_code','multiplicity'):
        require(np.array_equal(geometry[key], internal[key]), f'Internal alignment mismatch {key}')
    require(internal['trace_metrics'].shape == (n,7,6), 'Trace shape mismatch')
    require(np.isfinite(internal['trace_metrics']).all(), 'Nonfinite trace')
    require(internal['trace_metrics'].dtype == np.float32, 'Trace dtype mismatch')
    require(internal['trace_provenance'].shape == (n,), 'Trace provenance shape mismatch')
    require(np.count_nonzero(internal['trace_provenance']==1)==51560 and np.count_nonzero(internal['trace_provenance']==2)==673510, 'Trace provenance count mismatch')
    require(internal['stage_names'].tolist() == config['stage_names'], 'Stage order mismatch')
    require(internal['metric_names'].tolist() == config['metric_names'], 'Metric order mismatch')
    changed = codes != 0
    require(int(changed.sum()) == 25820, 'Changed count mismatch')
    for key in ('source_index','candidate_index'):
        require(np.array_equal(internal['patch_'+key], geometry[key][changed]), f'Patch alignment mismatch {key}')
    require(internal['patch_scores'].shape == (25820,7,35), 'Patch score shape mismatch')
    require(internal['patch_prediction'].shape == (25820,7), 'Patch prediction shape mismatch')
    require(np.isfinite(internal['patch_scores']).all(), 'Nonfinite patch scores')
    require(np.array_equal(internal['patch_scores'].argmax(axis=2), internal['patch_prediction']), 'Patch argmax mismatch')
    for key in ('patch_clean_class_margin','patch_true_class_margin'):
        require(internal[key].shape == (25820,7) and np.isfinite(internal[key]).all(), 'Invalid patch margin')
    for key, targets in [('patch_clean_class_margin',geometry['clean_prediction'][changed]), ('patch_true_class_margin',geometry['label'][changed])]:
        a=internal['patch_scores'].copy()
        target_score=np.take_along_axis(a,targets[:,None,None].repeat(7,axis=1),axis=2).squeeze(2)
        np.put_along_axis(a,targets[:,None,None].repeat(7,axis=1),-np.inf,axis=2)
        require(np.array_equal(target_score-a.max(axis=2),internal[key]), 'Patch margin reconstruction mismatch '+key)
    patched_clean=np.stack([clean[int(s)] for s in internal['patch_source_index']])
    for stage in (0,3,6):
        require(np.array_equal(internal['patch_scores'][:,stage],patched_clean), 'Positive-control scores not exactly clean')
    # All candidate specifications, source labels, horizons and multiplicities.
    source_rows=[]
    offset=0
    for manifest_row, stored in zip(manifest, stored_source_rows):
        sid=manifest_row['source_index']; x=validate_input(inputs[f'source_{sid:05d}'])
        require(x.dtype == np.uint16, 'Panel dtype mismatch')
        require(sid == stored['source_index'], 'Manifest and source table order mismatch')
        digest=hashlib.sha256(x.astype('<u2').tobytes(order='C')).hexdigest()
        require(digest == manifest_row['input_array_sha256'], 'Panel array hash mismatch')
        require(x.shape == (manifest_row['horizon_steps'],140) and int(x.sum()) == manifest_row['input_count'], 'Panel cardinality mismatch')
        count=manifest_row['unique_candidates']; sl=slice(offset,offset+count)
        require(np.all(geometry['source_index'][sl] == sid), 'Source segment mismatch')
        require(np.all(geometry['audit_rank'][sl] == manifest_row['audit_rank']), 'Audit rank mismatch')
        require(np.array_equal(geometry['candidate_index'][sl],np.arange(count)), 'Missing or reordered candidate')
        require(np.all(geometry['label'][sl] == manifest_row['label']), 'Source label mismatch')
        require(np.all(geometry['clean_prediction'][sl] == clean[sid].argmax()), 'Clean prediction mismatch')
        specs=enumerate_moves(x)
        for value,key in zip(specs,('from_bin','to_bin','feature','multiplicity')):
            require(np.array_equal(value,geometry[key][sl]), 'Candidate operator mismatch '+key)
        require(np.all(geometry['direction'][sl] == specs[1]-specs[0]), 'Direction mismatch')
        require(np.array_equal((specs[0]/max(1,len(x)-1)).astype(np.float32),geometry['time_normalized'][sl]), 'Time coordinate mismatch')
        require(np.array_equal((specs[2]/139).astype(np.float32),geometry['feature_normalized'][sl]), 'Feature coordinate mismatch')
        require(abs(float(np.sort(clean[sid])[-1]-np.sort(clean[sid])[-2])-stored['clean_margin']) < 1e-12, 'Clean margin mismatch')
        a={'label':np.array(manifest_row['label']), 'clean_prediction':np.array(clean[sid].argmax()),
           'predictions':geometry['candidate_prediction'][sl], 'from_t':specs[0], 'to_t':specs[1], 'feature':specs[2], 'multiplicity':specs[3],
           'candidate_clean_class_margin':geometry['candidate_clean_class_margin'][sl], 'candidate_true_class_margin':geometry['candidate_true_class_margin'][sl]}
        row=derive_source_geometry(stored,a)
        errors=compare_tree(row,stored,atol=1e-12)
        require(not errors, 'Recomputed source summary mismatch: '+str(errors[:3]))
        source_rows.append(row);offset+=count
    require(offset==n,'Candidate coverage mismatch')
    return dict(config=config, geometry=geometry, internal=internal, inputs=inputs, clean_scores=clean, manifest=manifest, source_rows=source_rows)


def total_counts(geometry: dict) -> dict:
    codes=geometry['transition_code']; weights=geometry['multiplicity'].astype(np.int64)
    result={'unique_candidates':len(codes),'event_weighted_candidates':int(weights.sum())}
    for code,name in enumerate(GROUPS):
        selected=codes==code
        result[name+'_unique']=int(selected.sum())
        result[name+'_event_weighted']=int(weights[selected].sum())
    result['class_changed_unique']=int((codes!=0).sum())
    result['class_changed_event_weighted']=int(weights[codes!=0].sum())
    return result


def panel_statistics(rows: list[dict], config: dict) -> dict:
    clean=[row for row in rows if row['clean_correct']]
    margins=np.array([row['clean_margin'] for row in clean], dtype=np.float64)
    rates=np.array([row['adverse_unique']/row['unique_candidates'] for row in clean], dtype=np.float64)
    labels=np.array([row['label'] for row in clean])
    groups=[np.flatnonzero(labels==label) for label in np.unique(labels)]
    rho=spearman_rho(margins,rates); reps=config['repetitions']
    rng=np.random.default_rng(config['margin_permutation_seed']); extreme=0
    for _ in range(reps):
        permuted=rates.copy()
        for group in groups:permuted[group]=rates[rng.permutation(group)]
        extreme += spearman_rho(margins,permuted) <= rho
    rng=np.random.default_rng(config['adverse_bootstrap_seed']); boot=[]
    for _ in range(reps):
        sampled=[]
        for group in groups:sampled.extend(rates[rng.choice(group,size=len(group),replace=True)].tolist())
        boot.append(float(np.mean(sampled)))
    positive=sorted((row for row in clean if row['adverse_unique']>0),key=lambda r:(-r['adverse_unique'],r['source_index']))
    cumulative=np.cumsum([row['adverse_unique'] for row in positive])/sum(row['adverse_unique'] for row in positive)
    return dict(clean_correct_sources=len(clean), adverse_sensitive_sources=len(positive),
        any_change_sources=sum(row['class_changed_unique']>0 for row in rows),
        mean_source_class_change_rate=float(np.mean([row['class_change_rate'] for row in rows])),
        mean_adverse_rate=float(rates.mean()), adverse_rate_ci95=np.percentile(boot,[2.5,97.5]).tolist(),
        spearman_rho=rho, within_label_one_sided_p=float((extreme+1)/(reps+1)),
        ranked_adverse_sources=[row['source_index'] for row in positive], cumulative_adverse_share=cumulative.tolist())


def independent_checks(bundle: dict, internal_stats: dict, panel_stats: dict) -> dict:
    """A second counting/statistics route, not a call back into the primary helpers."""
    from scipy.stats import spearmanr, false_discovery_control
    g=bundle['geometry']; counts=Counter(); weighted=Counter()
    for label,clean,candidate,multiplicity in zip(g['label'].tolist(),g['clean_prediction'].tolist(),g['candidate_prediction'].tolist(),g['multiplicity'].tolist()):
        if candidate==clean:kind='class_preserved'
        elif clean==label:kind='adverse'
        elif candidate==label:kind='corrective'
        else:kind='lateral'
        counts[kind]+=1;weighted[kind]+=multiplicity
    totals=total_counts(g)
    require(all(counts[k]==totals[k+'_unique'] and weighted[k]==totals[k+'_event_weighted'] for k in GROUPS),'Independent scalar partition mismatch')
    correct=[row for row in bundle['source_rows'] if row['clean_correct']]
    independent_rho=float(spearmanr([row['clean_margin'] for row in correct],[row['adverse_rate'] for row in correct]).statistic)
    require(abs(independent_rho-panel_stats['spearman_rho'])<1e-12,'Independent Spearman mismatch')
    inf=internal_stats['source_level_inference']; families=[]
    families.append(([inf['transition_vs_preserved'][group][stage] for group in GROUPS[1:] for stage in bundle['config']['stage_names']], 'fdr_bh_q_within_transition_vs_preserved_family'))
    families.append(([inf['global_vs_local'][group][f'block_{b}']['attention_minus_local_trace_divergence'] for group in GROUPS for b in (1,2)],'fdr_bh_q_within_global_vs_local_trace_family'))
    families.append(([inf['global_vs_local'][group][f'block_{b}']['attention_minus_local_clean_prediction_restoration'] for group in GROUPS[1:] for b in (1,2)],'fdr_bh_q_within_global_vs_local_patch_family'))
    for family,key in families:
        expected=false_discovery_control([row['two_sided_signflip_p'] for row in family],method='bh')
        require(np.allclose(expected,[row[key] for row in family],rtol=0,atol=1e-12),'Independent FDR mismatch')
    return {'status':'PASS','scalar_candidate_count':sum(counts.values()),'unique_partition':dict(counts),'weighted_partition':dict(weighted),'scipy_spearman_rho':independent_rho,'fdr_families_checked':[len(v[0]) for v in families]}


def export_tables(out: Path, bundle: dict, benchmarks: dict, totals: dict, panel: dict, internal_stats: dict) -> None:
    write_csv(out/'tables/benchmark.csv',[dict(split=name,**{k:v for k,v in row.items() if k in ('samples','correct','accuracy','balanced_accuracy','macro_f1','worst_class_recall')}) for name,row in benchmarks.items()])
    outcome=[]
    for name in [GROUPS[0],'class_changed',*GROUPS[1:]]:
        outcome.append(dict(outcome=name,unique_n=totals[name+'_unique'],unique_percent=100*totals[name+'_unique']/totals['unique_candidates'],event_weighted_n=totals[name+'_event_weighted'],event_weighted_percent=100*totals[name+'_event_weighted']/totals['event_weighted_candidates']))
    write_csv(out/'tables/outcome_partition.csv',outcome)
    write_csv(out/'tables/source_geometry.csv',bundle['source_rows'])
    write_csv(out/'tables/adverse_concentration.csv',[dict(rank=i+1,source_index=s,cumulative_percent=100*v) for i,(s,v) in enumerate(zip(panel['ranked_adverse_sources'],panel['cumulative_adverse_share']))])
    traces=[];patches=[];inf=internal_stats['source_level_inference']
    for stage in bundle['config']['stage_names']:
        for group in GROUPS[1:]:
            q=inf['transition_vs_preserved'][group][stage]
            traces.append(dict(boundary=stage,transition=group,mean_contrast=q['mean_contrast'],ci_low=q['source_bootstrap95'][0],ci_high=q['source_bootstrap95'][1],p=q['two_sided_signflip_p'],q=q['fdr_bh_q_within_transition_vs_preserved_family'],n_sources=q['n_sources']))
        p=inf['patch_restoration']['adverse'][stage]
        patches.append(dict(boundary=stage,candidate_pooled_percent=100*internal_stats['clean_activation_patch']['adverse'][stage]['restored_clean_prediction_rate_unique'],equal_source_mean_percent=100*p['mean_restored_clean_prediction_rate'],ci95_low_percent=100*p['source_bootstrap95_restored_clean_prediction_rate'][0],ci95_high_percent=100*p['source_bootstrap95_restored_clean_prediction_rate'][1]))
    write_csv(out/'tables/trace_contrasts.csv',traces);write_csv(out/'tables/adverse_patching.csv',patches)
    # Keep publication rounding explicit and generate files accepted by the supplied MATLAB code.
    label_map={'class_preserved':'Class-preserved','adverse':'Adverse','corrective':'Corrective','lateral':'Lateral'}
    fig2=[dict(outcome=label_map[name],unique_n=totals[name+'_unique'],event_weighted_n=totals[name+'_event_weighted']) for name in GROUPS]
    write_csv(out/'matlab/fig2_outcome_partition.csv',fig2)
    write_csv(out/'matlab/fig3_source_geometry.csv',[{**row,'clean_correct':int(row['clean_correct'])} for row in bundle['source_rows']])
    names={'stem':'Stem','attention_1':'Attention 1','local_1':'Local 1','block_1':'Block 1','attention_2':'Attention 2','local_2':'Local 2','block_2':'Block 2'}
    wide=[]
    for stage in bundle['config']['stage_names']:
        row={'boundary':names[stage]}
        for group in GROUPS[1:]:
            q=inf['transition_vs_preserved'][group][stage];row[group+'_contrast']=round(q['mean_contrast'],4);row[group+'_q']=round(q['fdr_bh_q_within_transition_vs_preserved_family'],4)
        wide.append(row)
    write_csv(out/'matlab/fig4_trace_contrasts.csv',wide)
    write_csv(out/'matlab/fig4_adverse_patching.csv',[{'boundary':names[p['boundary']],**{k:round(v,2) for k,v in p.items() if k!='boundary'}} for p in patches if p['boundary'].startswith(('attention','local'))])
    import shutil
    shutil.copyfile(bundle['root']/'matlab/make_ssc_manuscript_figures.m',out/'matlab/make_ssc_manuscript_figures.m')


def reproduce(root: Path, out: Path) -> dict:
    print('Checking every panel input, candidate record and intervention output...',flush=True)
    b=load_and_validate(root);b['root']=root
    benchmark_ref=read_json(root/'data/reference/benchmark_metrics.json'); benchmarks={}
    for name in ('validation','test'):
        a=load_npz(root/f'data/benchmark/{name}_predictions.npz')
        m=benchmark_metrics(a['labels'],a['predictions']);benchmarks[name]=m
        require(np.array_equal(m['confusion'],a['confusion']), 'Benchmark stored confusion mismatch')
        require(np.array_equal(a['source_indices'],np.arange(len(a['labels']))),'Benchmark order mismatch')
        if 'scores' in a:require(np.array_equal(a['scores'].argmax(axis=1),a['predictions']),'Benchmark score argmax mismatch')
        expected={k:benchmark_ref[name][k] for k in m if k in benchmark_ref[name]}
        observed={k:m[k] for k in expected}
        require(not compare_tree(observed,expected),'Benchmark metrics mismatch')
        require(m['correct'] == {'validation':8617,'test':17247}[name], 'Benchmark correct-count mismatch')
    totals=total_counts(b['geometry'])
    require(not compare_tree(totals,read_json(root/'data/reference/neighborhood_totals.json')), 'Neighborhood totals mismatch')
    _,geometry_stats=geometry_aggregates(b['geometry'],b['source_rows'])
    errors=compare_tree(geometry_stats,read_json(root/'data/reference/geometry_statistics.json'))
    require(not errors,'Geometry statistics mismatch '+str(errors[:3]))
    print('Recomputing source bootstraps, permutations and internal contrasts...',flush=True)
    panel=panel_statistics(b['source_rows'],b['config'])
    internal_stats=recompute_internal(b['geometry'],b['internal'],b['clean_scores'])
    agreement=compare_internal_statistics(internal_stats,read_json(root/'data/reference/internal_statistics.json'))
    write_json(out/'numerical_agreement.json',agreement)
    require(agreement['status']=='PASS','Internal statistics mismatch '+str(agreement['errors'][:5]))
    print('Running independent scalar counts, Spearman and FDR checks...',flush=True)
    second=independent_checks(b,internal_stats,panel)
    trace=b['internal']['trace_metrics'];preserved=b['geometry']['transition_code']==0
    changed_internal=int(np.count_nonzero(trace[preserved,6,2]>0))
    require(changed_internal==617941,'Preserved internal-change count mismatch')
    require(abs(panel['spearman_rho']-(-0.5688))<0.00005,'Rounded manuscript correlation mismatch')
    require(abs(panel['mean_adverse_rate']*100-0.9596)<0.00005,'Rounded manuscript adverse mean mismatch')
    require(panel['clean_correct_sources']==84 and panel['adverse_sensitive_sources']==13 and panel['any_change_sources']==25,'Source incidence mismatch')
    write_json(out/'statistics/benchmark.json',benchmarks)
    write_json(out/'statistics/outcomes.json',totals)
    write_json(out/'statistics/panel.json',panel)
    write_json(out/'statistics/geometry.json',geometry_stats)
    write_json(out/'statistics/internal.json',internal_stats)
    export_tables(out,b,benchmarks,totals,panel,internal_stats)
    report={'status':'PASS','scope':'Statistics regenerated from released arrays, not new neural-network inference.',
        'candidates_checked':725070,'source_records_checked':100,'trace_metric_values_checked':int(trace.size),
        'patched_forward_outputs_checked':25820*7,'reference_tolerances':{'counts_and_predictions':'exact','p_values_q_values_rates_float64':1e-12,'float32_derived_means_and_intervals':1e-6},
        'max_internal_summary_absolute_difference':agreement['max_absolute_difference'],
        'independent_checks':second,'preserved_with_changed_final_activation':changed_internal,
        'new_model_inference_performed':False,'new_training_performed':False,'raw_official_test_opened':False,
        'matlab_rendered':False,'loss_recomputed':False,
        'benchmark_note':'All classification metrics were recomputed. Cross-entropy loss is outside the manuscript tables and is not included in this CPU-only summary regeneration.'}
    write_json(out/'validation.json',report)
    return report
