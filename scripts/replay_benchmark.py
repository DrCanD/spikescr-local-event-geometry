"""Explicit benchmark replay using the registered batch size of 256.

Never used for selecting weights, thresholds or audit examples. Raw dataset
bytes must be supplied explicitly and match the recorded split hash.
"""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ssc_geometry.io import read_json,load_npz,write_json,sha256_file,fresh_output
from ssc_geometry.integrity import verify_integrity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',choices=('validation','test'),required=True)
    p.add_argument('--h5',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--upstream',type=Path,default=ROOT/'.cache/upstream')
    p.add_argument('--device',default='cuda')
    p.add_argument('--allow-official-test-replay',action='store_true')
    p.add_argument('--allow-nonreference-environment',action='store_true')
    a=p.parse_args();verify_integrity(ROOT)
    if a.split=='test' and not a.allow_official_test_replay:
        raise ValueError('Official test replay requires --allow-official-test-replay; the default does not access it.')
    config=read_json(ROOT/'configs/experiment.json')
    expected=config['validation_h5_sha256'] if a.split=='validation' else config['official_test_h5_sha256']
    if sha256_file(a.h5)!=expected:raise ValueError('Raw input file does not match the recorded split hash')
    out=fresh_output(ROOT,a.out)
    try:
        import h5py,numpy as np,torch
        from ssc_geometry.core import transform_events
        from ssc_geometry.analysis import benchmark_metrics
        from ssc_geometry.upstream import load_model
        from ssc_geometry import _forward_kernel as fk
        model,device,_,env=load_model(ROOT,a.upstream,a.device,a.allow_nonreference_environment)
        write_json(out/'environment.json',env)
        scores=[]; labels=[]; lengths=[]
        with h5py.File(a.h5,'r') as h:
            n=len(h['labels']);expected_n=9981 if a.split=='validation' else 20382
            if n!=expected_n:raise ValueError('Wrong split cardinality')
            for start in range(0,n,256):
                xs=[]
                for i in range(start,min(start+256,n)):
                    x=transform_events(h['spikes/times'][i],h['spikes/units'][i]);xs.append(torch.from_numpy(x.astype(np.float32)));labels.append(int(h['labels'][i]));lengths.append(len(x))
                batch=torch.nn.utils.rnn.pad_sequence(xs,batch_first=True,padding_value=0.).to(device)
                lens=torch.tensor([len(x) for x in xs],device=device)
                fk.reset_spiking_state(model)
                try:
                    with torch.no_grad():value=fk.upstream_softmax_sum(model(batch,fk.attention_mask_from_lengths(lens,batch.shape[1])))
                    if not torch.isfinite(value).all():raise ValueError('Nonfinite benchmark output')
                    scores.append(value.cpu().numpy())
                finally:fk.reset_spiking_state(model)
        score=np.concatenate(scores);label=np.asarray(labels);prediction=score.argmax(axis=1)
        ref=load_npz(ROOT/f'data/benchmark/{a.split}_predictions.npz')
        differences=int(np.count_nonzero(prediction!=ref['predictions']))
        if not np.array_equal(label,ref['labels']):raise ValueError('Benchmark labels/order mismatch')
        score_diff=float(np.max(np.abs(score-ref['scores']))) if 'scores' in ref else None
        metrics=benchmark_metrics(label,prediction)
        np.savez_compressed(out/'predictions.npz',source_indices=np.arange(n),labels=label,predictions=prediction,scores=score,lengths=np.array(lengths),confusion=np.asarray(metrics['confusion']))
        report={'status':'PASS' if differences==0 and (score_diff is None or score_diff<=1e-5) else 'FAIL',
            'split':a.split,'samples':n,'prediction_differences':differences,'score_max_absolute_difference':score_diff,
            'batch_size':256,'new_inference_performed':True,'raw_official_test_opened':a.split=='test','selection_or_tuning_performed':False,
            'environment':env,'metrics':metrics}
        write_json(out/'benchmark_replay.json',report)
        print(json.dumps(report,indent=2))
        if report['status']!='PASS':raise ValueError('Benchmark does not conform to the committed reference; reference records were not changed')
    except Exception as exc:
        write_json(out/'FAILED.json',{'status':'FAIL','error':str(exc),'exception':type(exc).__name__});raise
if __name__=='__main__':main()
