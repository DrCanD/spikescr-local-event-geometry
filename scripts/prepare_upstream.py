"""Download and hash-check pinned upstream source into a local ignored cache."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ssc_geometry.integrity import verify_integrity
from ssc_geometry.upstream import prepare_source

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination',type=Path,default=ROOT/'.cache/upstream')
    a=p.parse_args()
    verify_integrity(ROOT)
    from ssc_geometry.io import assert_output_safe
    assert_output_safe(ROOT,a.destination)
    print(json.dumps(prepare_source(ROOT,a.destination),indent=2))
if __name__=='__main__':main()
