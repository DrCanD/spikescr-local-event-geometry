"""Command-line entry points. Every expensive or network operation is explicit."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from .io import repository_root, fresh_output, write_json


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description='Reproduce the frozen SpikeSCR local event audit')
    parser.add_argument('--root',type=Path,help='Unpacked repository directory')
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('verify',help='Verify all release checksums and immutable paths')
    sub.add_parser('checkpoint',help='Safely inspect and hash the frozen checkpoint; requires torch')
    p=sub.add_parser('reproduce',help='Recompute tables and statistics from released predictions and traces')
    p.add_argument('--out',type=Path,required=True)
    p=sub.add_parser('raw-validation',help='Compare both preprocessing implementations with released panel inputs')
    p.add_argument('--h5',type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        root=repository_root(args.root)
        from .integrity import verify_integrity, verify_checkpoint, verify_raw_validation
        integrity=verify_integrity(root)
        if args.command=='verify':report=integrity
        elif args.command=='checkpoint':report=verify_checkpoint(root)
        elif args.command=='raw-validation':report=verify_raw_validation(root,args.h5)
        else:
            from .analysis import reproduce
            out=fresh_output(root,args.out)
            try:report=reproduce(root,out)
            except Exception as exc:
                write_json(out/'FAILED.json',{'status':'FAIL','error':str(exc),'exception':type(exc).__name__})
                raise
            write_json(out/'integrity.json',integrity)
        print(json.dumps(report,indent=2,sort_keys=True))
        return 0
    except Exception as exc:
        print(f'FAIL: {type(exc).__name__}: {exc}',file=sys.stderr)
        return 1


if __name__=='__main__':raise SystemExit(main())
