"""Input, perturbation and transition contracts used by the released audit."""
from __future__ import annotations
import numbers
import numpy as np
from ._kernel import exact_event_transform, candidate_specification


def validate_input(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != 140:
        raise ValueError('Expected nonempty [time,140] tensor')
    if x.dtype.kind not in 'iuf' or not np.isfinite(x).all():
        raise ValueError('Counts must be finite numbers')
    if np.any(x < 0) or np.any(x != np.floor(x)) or np.any(x > 65535):
        raise ValueError('Counts must be nonnegative integers no greater than 65535')
    return x


def transform_events(times: np.ndarray, units: np.ndarray) -> np.ndarray:
    times, units = np.asarray(times), np.asarray(units)
    if times.ndim != 1 or units.ndim != 1 or times.shape != units.shape:
        raise ValueError('Times and units must be equally sized vectors')
    if times.dtype.kind not in 'iuf' or units.dtype.kind not in 'iuf':
        raise ValueError('Numeric events required')
    if not np.isfinite(times).all() or not np.isfinite(units).all():
        raise ValueError('Nonfinite event')
    if np.any(units != np.floor(units)):
        raise ValueError('Unit IDs must be integers')
    return exact_event_transform(times, units, 700, 5, 5)[0]


def enumerate_moves(x: np.ndarray) -> tuple[np.ndarray, ...]:
    return candidate_specification(validate_input(x))


def move_one_count(x: np.ndarray, from_bin: int, to_bin: int, feature: int) -> np.ndarray:
    x = validate_input(x)
    if any(not isinstance(v, numbers.Integral) or isinstance(v, (bool, np.bool_)) for v in (from_bin, to_bin, feature)):
        raise ValueError('Move indices must be integers')
    if not (0 <= from_bin < len(x) and 0 <= to_bin < len(x) and 0 <= feature < 140):
        raise ValueError('Move outside the fixed horizon or feature range')
    if abs(int(to_bin) - int(from_bin)) != 1 or x[from_bin, feature] < 1:
        raise ValueError('Move must transfer one available count to an adjacent bin')
    if x[to_bin, feature] >= 65535:
        raise OverflowError('Destination exceeds the stored count representation')
    result = x.astype(np.int32, copy=True)
    result[from_bin, feature] -= 1
    result[to_bin, feature] += 1
    return result


def classify_transitions(labels: np.ndarray, clean: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    labels, clean, candidate = map(np.asarray, (labels, clean, candidate))
    if labels.shape != clean.shape or labels.shape != candidate.shape:
        raise ValueError('Label and prediction shapes disagree')
    if any(a.dtype.kind not in 'iu' or np.any((a < 0) | (a >= 35)) for a in (labels, clean, candidate)):
        raise ValueError('Class IDs must be integers in [0,34]')
    changed = candidate != clean
    out = np.zeros(labels.shape, dtype=np.uint8)
    out[changed & (clean == labels)] = 1
    out[changed & (clean != labels) & (candidate == labels)] = 2
    out[changed & (clean != labels) & (candidate != labels)] = 3
    return out
