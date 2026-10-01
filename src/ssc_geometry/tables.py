"""Manuscript cross-check: every number printed in the manuscript tables and text is looked up in the
regenerated results and compared with its printed value (``data/reference/expected_values.json``).

A cell passes when the regenerated value, rounded as the manuscript prints it, equals the printed
value (counts exactly; decimals within half a unit of the last printed digit).
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Any
from . import paths
from .io import read_json, write_json, write_csv

_SELECTOR = re.compile(r'^([^\[]+)(\[(.*)\])?$')


def resolve(results: Any, path: str) -> Any:
    """Follow a dotted path; ``name[3]`` indexes a list, ``name[key=value,...]`` selects the unique matching dict."""
    node = results
    for step in path.split('.'):
        match = _SELECTOR.match(step)
        if match is None:
            raise KeyError(path)
        name, _, selector = match.groups()
        node = node[name] if name else node
        if selector is None:
            continue
        if selector.isdigit():
            node = node[int(selector)]
            continue
        conditions = dict(part.split('=', 1) for part in selector.split(','))
        matches = [item for item in node if all(str(item.get(k)) == v for k, v in conditions.items())]
        if len(matches) != 1:
            raise KeyError(f'{path}: {len(matches)} matches for [{selector}]')
        node = matches[0]
    return node


def check_cell(results: dict, cell: dict) -> dict:
    expected = cell['value']
    try:
        raw = resolve(results, cell['path'])
    except (KeyError, IndexError, TypeError) as exc:
        return dict(**cell, computed=None, status='FAIL', reason=f'missing: {exc}')
    if isinstance(expected, bool) or isinstance(raw, bool):
        return dict(**cell, computed=raw, status='PASS' if bool(raw) == bool(expected) else 'FAIL')
    if isinstance(expected, str):
        return dict(**cell, computed=raw, status='PASS' if str(raw) == expected else 'FAIL')
    if isinstance(expected, list):
        computed = [float(v) * cell.get('scale', 1) for v in raw]
        tolerance = 0.5 * 10 ** (-cell.get('decimals', 0)) + 1e-9
        ok = len(computed) == len(expected) and all(abs(c - e) <= tolerance for c, e in zip(computed, expected))
        return dict(**cell, computed=computed, status='PASS' if ok else 'FAIL', tolerance=tolerance)
    computed = float(raw) * cell.get('scale', 1)
    decimals = cell.get('decimals', 0)
    tolerance = 0.5 * 10 ** (-decimals) + 1e-9 if decimals else 0.0
    ok = abs(computed - expected) <= tolerance if decimals else int(round(computed)) == expected and abs(computed - round(computed)) < 1e-9
    return dict(**cell, computed=computed, difference=computed - expected, tolerance=tolerance, status='PASS' if ok else 'FAIL')


def cross_check(root: Path, results: dict, out: Path) -> dict:
    """Compare every manuscript cell with the regenerated results; write report.json and report.csv."""
    spec = read_json(root / paths.EXPECTED_VALUES)
    cells = [check_cell(results, cell) for cell in spec['cells']]
    failures = [c for c in cells if c['status'] != 'PASS']
    report = dict(status='PASS' if not failures else 'FAIL', manuscript=spec['manuscript'], cells_checked=len(cells),
                  cells_failed=len(failures), failed_ids=[c['id'] for c in failures],
                  rule='counts exact; printed decimals within half a unit of the last digit; cells of kind summary are read from data/execution/summary.json, not recomputed',
                  cells=cells)
    write_json(out / 'report.json', report)
    write_csv(out / 'report.csv', [dict(id=c['id'], location=c['location'], expected=c['value'], computed=c.get('computed'),
                                        status=c['status'], kind=c.get('kind', 'recomputed')) for c in cells])
    return report
