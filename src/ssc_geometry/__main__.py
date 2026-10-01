"""Command-line entry points.

    python -m ssc_geometry verify                      integrity of every released file
    python -m ssc_geometry paper     --out DIR         every manuscript table + report.json cross-check
    python -m ssc_geometry census    --out DIR         Tables 3, 5-8 and figure data (§4.1-4.2, 4.4-4.6)
    python -m ssc_geometry search    --out DIR         Table 4 and the cutoff sweep (§4.3, S2)
    python -m ssc_geometry replicas  --out DIR         Table 9 and the paired bootstrap (§4.7)
    python -m ssc_geometry execution --out DIR         Tables 2 and A.1-A.3 (§4.1, 4.8, Appendix A)
    python -m ssc_geometry checkpoint                  hash and inspect the frozen checkpoint (needs torch)
    python -m ssc_geometry raw-validation --h5 FILE    rebuild the panel inputs from the official validation HDF5

None of the table commands executes the model or needs a GPU. Output directories are created fresh;
an existing directory is never overwritten.
"""
from __future__ import annotations
import argparse
import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path
from . import __version__
from .io import repository_root, fresh_output, write_json, sha256_file
from . import paths

TABLE_COMMANDS = ('census', 'search', 'replicas', 'execution')


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    except Exception:
        return None


def _run_table_command(name: str, root: Path, out: Path) -> dict:
    if name == 'census':
        from .census import run
    elif name == 'search':
        from .search import run
    elif name == 'replicas':
        from .replicas import run
    else:
        from .execution import run
    return run(root, out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='python -m ssc_geometry', description='Regenerate the manuscript tables from the released records.')
    parser.add_argument('--root', type=Path, help='repository directory (default: the checkout this package was installed from)')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('verify', help='check every released file against provenance/manifest.json')
    for name, help_text in [('paper', 'all tables and figure data, plus report.json comparing every manuscript cell'),
                            ('census', 'Tables 3, 5, 6, 7, 8 and figure data'), ('search', 'Table 4, prefix costs, cutoff sweep'),
                            ('replicas', 'Table 9 and the paired replica bootstrap'), ('execution', 'Tables 2, A.1, A.2, A.3')]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument('--out', type=Path, required=True, help='new output directory')
    sub.add_parser('checkpoint', help='hash and inspect the frozen checkpoint (requires torch; no forward pass)')
    p = sub.add_parser('raw-validation', help='rebuild the 100 panel inputs from the official validation HDF5')
    p.add_argument('--h5', type=Path, required=True, help='path to ssc_valid.h5 (hash-checked before reading)')
    args = parser.parse_args(argv)
    try:
        root = repository_root(args.root)
        from .integrity import verify_integrity, verify_checkpoint, verify_raw_validation
        integrity = verify_integrity(root)
        if args.command == 'verify':
            report = integrity
        elif args.command == 'checkpoint':
            report = verify_checkpoint(root)
        elif args.command == 'raw-validation':
            report = verify_raw_validation(root, args.h5)
        else:
            out = fresh_output(root, args.out)
            started = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds')
            manifest = dict(command=args.command, package_version=__version__, repository_commit=_git_commit(root),
                            integrity_manifest_sha256=sha256_file(root / paths.MANIFEST), started_at=started, python=sys.version.split()[0])
            write_json(out / 'manifest.json', manifest)
            try:
                if args.command == 'paper':
                    results = {}
                    for name in TABLE_COMMANDS:
                        print(f'== {name}', flush=True)
                        results[name] = _run_table_command(name, root, out)
                    from .tables import cross_check
                    print('== manuscript cross-check', flush=True)
                    check = cross_check(root, results, out)
                    report = dict(status=check['status'], cells_checked=check['cells_checked'], cells_failed=check['cells_failed'],
                                  failed_ids=check['failed_ids'], commands={name: results[name]['status'] for name in TABLE_COMMANDS},
                                  output=str(out), new_model_inference=False)
                else:
                    full = _run_table_command(args.command, root, out)
                    report = dict(status=full['status'], command=args.command, output=str(out), new_model_inference=False)
            except Exception as exc:
                write_json(out / 'FAILED.json', {'status': 'FAIL', 'error': str(exc), 'exception': type(exc).__name__})
                raise
            manifest['finished_at'] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds')
            manifest['status'] = report['status']
            write_json(out / 'manifest.json', manifest)
            write_json(out / 'integrity.json', integrity)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report.get('status', 'PASS') == 'PASS' else 1
    except Exception as exc:
        print(f'FAIL: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
