"""Bounded, explicit data access. Nothing is downloaded or inferred implicitly."""
from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import numpy as np


def repository_root(root: str | Path | None = None) -> Path:
    if root is not None:
        candidate = Path(root).expanduser().resolve()
    else:
        candidate = Path(__file__).resolve().parents[2]
    if not (candidate / 'configs/experiment.json').is_file():
        raise FileNotFoundError('Repository data not found. Supply --root with the unpacked repository path.')
    return candidate


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    try:
        with temporary.open('w', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def read_numeric_csv(path: str | Path) -> list[dict]:
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in row.items():
            if value in ('True', 'False'):
                row[key] = value == 'True'
            else:
                try:
                    row[key] = int(value)
                except (ValueError, TypeError):
                    try:
                        row[key] = float(value)
                    except (ValueError, TypeError):
                        pass
        if 'clean_correct' in row:
            if row['clean_correct'] not in (0, 1, False, True):
                raise ValueError('clean_correct must be 0/1 or True/False')
            row['clean_correct'] = bool(row['clean_correct'])
    return rows


def write_csv(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError('Cannot infer schema for an empty CSV')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def assert_output_safe(root: Path, path: str | Path) -> Path:
    """Reject any output that could overwrite immutable repository contents."""
    root=root.resolve();result=Path(path).expanduser().resolve()
    protected=('data','configs','src','tests','checksums','docs','validation','scripts','.git','.github')
    if result==root or result in root.parents:
        raise ValueError('Output cannot be the repository or its parent')
    for part in protected:
        target=root/part
        if result==target or target in result.parents or result in target.parents:
            raise ValueError('Output overlaps protected repository files')
    return result


def fresh_output(root: Path, path: str | Path) -> Path:
    """Never overwrite references or an existing result folder."""
    result=assert_output_safe(root,path)
    if result.exists():
        raise FileExistsError('Output exists. Select a new output directory; results are never silently overwritten.')
    result.mkdir(parents=True)
    return result


def compare_tree(actual: Any, expected: Any, path: str = '', atol: float = 1e-12) -> list[str]:
    """Compare all recorded leaves, not only a handful of headline numbers."""
    errors = []
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            return [f'{path}: key mismatch']
        for key, value in expected.items():
            errors.extend(compare_tree(actual[key], value, f'{path}/{key}', atol))
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return [f'{path}: list length/type mismatch']
        for index, value in enumerate(expected):
            errors.extend(compare_tree(actual[index], value, f'{path}/{index}', atol))
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        if not isinstance(actual, (float, int)) or not np.isfinite(actual) or abs(actual - expected) > atol:
            errors.append(f'{path}: {actual!r} != {expected!r}')
    elif actual != expected:
        errors.append(f'{path}: {actual!r} != {expected!r}')
    return errors


def compare_internal_statistics(actual: dict, expected: dict) -> dict:
    """Separate float32 reduction portability from exact integer and p-value checks.

    The 1e-6 tolerance applies only to float32-derived descriptive means and
    their confidence limits. It is not used for predictions, p-values, q-values,
    patch recovery rates, file hashes, or candidate identity.
    """
    differences=[];errors=[]
    def visit(a,b,path):
        if isinstance(b,dict):
            if not isinstance(a,dict) or a.keys()!=b.keys():
                errors.append(path+': keys');return
            for k,v in b.items():visit(a[k],v,path+'/'+k)
        elif isinstance(b,list):
            if not isinstance(a,list) or len(a)!=len(b):
                errors.append(path+': length');return
            for i,v in enumerate(b):visit(a[i],v,path+'/'+str(i))
        elif isinstance(b,float):
            derived=(path.startswith('/trace_group_means_unique/') or
              ((path.startswith('/source_level_inference/transition_vs_preserved/') or
                '/attention_minus_local_trace_divergence/' in path) and
               ('/mean_contrast' in path or '/source_bootstrap95/' in path)))
            tol=1e-6 if derived else 1e-12
            if not isinstance(a,(float,int)) or not np.isfinite(a) or abs(a-b)>tol:
                errors.append(path+': numerical mismatch')
            elif a!=b:
                differences.append(dict(path=path,computed=a,recorded=b,absolute_difference=abs(a-b),absolute_tolerance=tol))
        elif a!=b:errors.append(path+': value')
    visit(actual,expected,'')
    return dict(status='PASS' if not errors else 'FAIL',errors=errors,
                differences=differences,max_absolute_difference=max((d['absolute_difference'] for d in differences),default=0),
                note='Float32 reduction results can differ across NumPy builds and CPU architectures; they are not described as bit-identical.')
