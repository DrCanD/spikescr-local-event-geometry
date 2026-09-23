"""Run explicit preflight or complete singleton inference against released records."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ssc_geometry.io import fresh_output,write_json
from ssc_geometry.integrity import verify_integrity
from ssc_geometry.inference import run_audit

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('preflight','full'),default='preflight')
    p.add_argument('--upstream',type=Path,default=ROOT/'.cache/upstream')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--allow-nonreference-environment',action='store_true')
    p.add_argument('--sources',type=int,nargs='+')
    p.add_argument('--max-candidates',type=int)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();verify_integrity(ROOT)
    if a.resume:
        out=a.out.resolve()
        # Same protection checks as a fresh run, without creating or deleting any files.
        from ssc_geometry.io import assert_output_safe
        assert_output_safe(ROOT,out)
        if not (out/'run_manifest.json').is_file():raise ValueError('No resumable manifest at output')
    else:out=fresh_output(ROOT,a.out)
    try:
        report=run_audit(ROOT,a.upstream,out,a.mode,a.device,a.allow_nonreference_environment,a.sources,a.max_candidates,a.resume)
        print(json.dumps(report,indent=2))
    except Exception as exc:
        if not (out/'FAILED.json').exists():write_json(out/'FAILED.json',{'status':'FAIL','exception':type(exc).__name__,'error':str(exc)})
        raise
if __name__=='__main__':main()
