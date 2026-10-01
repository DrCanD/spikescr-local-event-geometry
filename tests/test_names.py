"""No internal experiment labels, machine-specific paths or packaging-history vocabulary in the release.

Every tracked text file, every string array inside the .npz archives and every JSON string value is
scanned; packaging-history vocabulary (recorded, historical, revision) is additionally excluded from the
user-facing prose (README, docs, citation). SHA-256 digests are ignored. Two documented exemptions: provenance/records.json may name
original file paths in its ``original_path`` fields (the audit trail of the restructuring), and the
four contracts under provenance/contracts/ are kept verbatim because their hash is stored inside the
output arrays (their internal version labels are listed in provenance/records.json).
"""
import json
import re
import subprocess
import unittest
from pathlib import Path
import numpy as np
from ssc_geometry.io import repository_root

ROOT = repository_root()
LABELS = re.compile(r'\bEB\d+|\bH[1-5]_|[Pp]reregist|SSC_Experiments|GPU_Execution_Control|Mechanism_and_Device|SIPT|TIB_|WP2B|wp2b|Event_Geometry|audit_bundle')
PATHS = re.compile(r'/content/|MyDrive|[A-Za-z]:\\|/home/|/Users/|/tmp/|Colab')
DATES = re.compile(r'_2026\d{4}|2026092\d|2026093[01]')           # date-stamped names and version labels
VOCABULARY = re.compile(r'\brevision\b|\brecorded\b|\bhistorical\b|byte-preserved')  # user-facing prose only
SHA = re.compile(r'\b[0-9a-f]{40,64}\b')
PROSE = ('README.md', 'CITATION.cff', 'LICENSE_NOTICE.md')


def tracked_files():
    listing = subprocess.run(['git', '-C', str(ROOT), 'ls-files', '-z'], capture_output=True, check=True).stdout.decode()
    return [ROOT / name for name in listing.split('\0') if name]


def is_verbatim_contract(path: Path) -> bool:
    return path.parent == ROOT / 'provenance/contracts'


def json_strings(node, trail=''):
    if isinstance(node, dict):
        for k, v in node.items():
            yield trail + '/' + str(k), str(k)
            yield from json_strings(v, trail + '/' + str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from json_strings(v, f'{trail}/{i}')
    elif isinstance(node, str):
        yield trail, node


class ForbiddenStringTests(unittest.TestCase):
    def test_file_names(self):
        bad = [p.relative_to(ROOT).as_posix() for p in tracked_files() if LABELS.search(p.name) or DATES.search(p.name) or VOCABULARY.search(p.name)]
        self.assertEqual(bad, [])

    def test_text_files(self):
        hits = {}
        for path in tracked_files():
            if path.suffix in ('.npz', '.pt') or not path.is_file() or path.name == 'test_names.py':
                continue
            relative = path.relative_to(ROOT).as_posix()
            text = SHA.sub('', path.read_text(encoding='utf-8', errors='replace'))
            found = set(PATHS.findall(text))
            if path.suffix == '.json':
                strings = list(json_strings(json.loads(text if path.name != 'records.json' else path.read_text(encoding='utf-8'))))
                for trail, value in strings:
                    if 'original_path' in trail:
                        continue
                    if not is_verbatim_contract(path):
                        found.update(LABELS.findall(value)); found.update(DATES.findall(value))
            else:
                found.update(LABELS.findall(text)); found.update(DATES.findall(text))
            if path.name in PROSE or (path.suffix == '.md' and path.parent == ROOT / 'docs'):
                found.update(VOCABULARY.findall(text))
            if found:
                hits[relative] = sorted(found)
        self.assertEqual(hits, {})

    def test_npz_keys_and_string_arrays(self):
        hits = {}
        for path in tracked_files():
            if path.suffix != '.npz':
                continue
            with np.load(path, allow_pickle=False) as archive:
                for key in archive.files:
                    found = set(LABELS.findall(key)) | set(DATES.findall(key))
                    value = archive[key]
                    if value.dtype.kind in 'US':
                        for item in np.atleast_1d(value).ravel().tolist():
                            text = SHA.sub('', str(item))
                            found.update(LABELS.findall(text)); found.update(PATHS.findall(text))
                    if found:
                        hits[f'{path.relative_to(ROOT).as_posix()}:{key}'] = sorted(found)
        self.assertEqual(hits, {})

    def test_json_values_hold_no_file_system_paths(self):
        absolute = re.compile(r'^(/|[A-Za-z]:\\)\S*\.(h5|pt|npz|json|csv|py|txt|zip)$')
        hits = []
        for path in tracked_files():
            if path.suffix != '.json':
                continue
            for trail, value in json_strings(json.loads(path.read_text(encoding='utf-8'))):
                if absolute.match(value) and 'original_path' not in trail:
                    hits.append(f'{path.relative_to(ROOT).as_posix()}:{trail}')
        self.assertEqual(hits, [])

    def test_records_list_the_verbatim_contract_labels(self):
        records = json.loads((ROOT / 'provenance/records.json').read_text(encoding='utf-8'))
        kept = {entry['path'] for entry in records['verbatim_contracts']}
        self.assertEqual(kept, {p.relative_to(ROOT).as_posix() for p in (ROOT / 'provenance/contracts').glob('*.json')})


if __name__ == '__main__':
    unittest.main()
