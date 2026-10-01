"""Maintainer tool: regenerate provenance/manifest.json and provenance/SHA256SUMS.txt.

Every file tracked by git (except the two manifest files themselves) is hashed. Run it after any
committed change, then commit the two files; `python -m ssc_geometry verify` checks against them.
"""
from pathlib import Path
import argparse, json, subprocess, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ssc_geometry.io import sha256_file
from ssc_geometry import paths

EXCLUDED = {paths.MANIFEST, paths.SHA256SUMS}


def tracked_files(root: Path) -> list[str]:
    listing = subprocess.run(['git', '-C', str(root), 'ls-files', '-z'], capture_output=True, check=True).stdout
    return sorted(name for name in listing.decode('utf-8').split('\0') if name and name not in EXCLUDED)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check', action='store_true', help='only report whether the manifest is current (exit 1 if not)')
    a = p.parse_args()
    files = {name: sha256_file(ROOT / name) for name in tracked_files(ROOT) if (ROOT / name).is_file()}
    manifest = {'schema_version': 1, 'files': files}
    sums = ''.join(f'{digest}  {name}\n' for name, digest in files.items())
    current = (ROOT / paths.MANIFEST).is_file() and json.loads((ROOT / paths.MANIFEST).read_text(encoding='utf-8')) == manifest \
        and (ROOT / paths.SHA256SUMS).read_text(encoding='utf-8') == sums
    if a.check:
        print('manifest current' if current else 'manifest out of date')
        return 0 if current else 1
    (ROOT / paths.MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    (ROOT / paths.SHA256SUMS).write_text(sums, encoding='utf-8')
    print(f'{len(files)} files hashed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
