"""Input, perturbation and transition contracts used by the released audit."""
from __future__ import annotations
import math
import numbers
import numpy as np


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


# ---------------------------------------------------------------- event transform and candidate operator
def exact_event_transform(
    times_seconds: np.ndarray,
    units: np.ndarray,
    input_channels: int,
    spatial_bin: int,
    time_step_ms: int,
) -> tuple[np.ndarray, dict[str, int]]:
    times = np.asarray(times_seconds, dtype=np.float64)
    units = np.asarray(units, dtype=np.int64)
    if times.size != units.size:
        raise RuntimeError("times/units length mismatch")
    if units.size and (int(units.min()) < 0 or int(units.max()) >= input_channels):
        raise RuntimeError(f"Unit ID outside verified range 0..{input_channels - 1}")
    features = input_channels // spatial_bin
    if times.size == 0:
        return np.zeros((1, features), dtype=np.uint16), {
            "events": 0,
            "unit_min": input_channels,
            "unit_max": -1,
        }
    if np.any(np.diff(times) < 0):
        raise RuntimeError("Non-monotonic event times")
    times_ms = (times - times[0]) * 1000.0
    frames_num = max(1, int(math.ceil(float(times_ms[-1]) / float(time_step_ms))))
    frame_ids = np.floor_divide(times_ms, time_step_ms).astype(np.int64)
    frame_ids = np.minimum(frame_ids, frames_num - 1)
    feature_ids = units // spatial_bin
    flat = frame_ids * features + feature_ids
    counts = np.bincount(flat, minlength=frames_num * features).reshape(frames_num, features)
    if int(counts.max(initial=0)) > np.iinfo(np.uint16).max:
        raise OverflowError("SSC frame count exceeded uint16 range")
    return counts.astype(np.uint16, copy=False), {
        "events": int(units.size),
        "unit_min": int(units.min()),
        "unit_max": int(units.max()),
    }

def literal_event_transform(
    times_seconds: np.ndarray,
    units: np.ndarray,
    input_channels: int,
    spatial_bin: int,
    time_step_ms: int,
) -> np.ndarray:
    times = np.asarray(times_seconds, dtype=np.float64)
    units = np.asarray(units, dtype=np.int64)
    features = input_channels // spatial_bin
    if times.size == 0:
        return np.zeros((1, features), dtype=np.uint16)
    times_ms = (times - times[0]) * 1000.0
    frames_num = max(1, int(math.ceil(float(times_ms[-1]) / float(time_step_ms))))
    output = np.zeros((frames_num, features), dtype=np.uint32)
    frame_ids = np.floor_divide(times_ms, time_step_ms).astype(np.int64)
    for i in range(units.size):
        frame = min(int(frame_ids[i]), frames_num - 1)
        output[frame, int(units[i]) // spatial_bin] += 1
    return output.astype(np.uint16)

def candidate_specification(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1] != 140:
        raise ValueError(f"Expected [T,140] transformed source, got {x.shape}")
    positive_t, positive_f = np.nonzero(x > 0)
    from_t: list[int] = []
    to_t: list[int] = []
    features: list[int] = []
    multiplicity: list[int] = []
    horizon = int(x.shape[0])
    for t, feature in zip(positive_t.tolist(), positive_f.tolist()):
        count = int(x[t, feature])
        if t > 0:
            from_t.append(t); to_t.append(t - 1); features.append(feature); multiplicity.append(count)
        if t + 1 < horizon:
            from_t.append(t); to_t.append(t + 1); features.append(feature); multiplicity.append(count)
    return (
        np.asarray(from_t, dtype=np.int32),
        np.asarray(to_t, dtype=np.int32),
        np.asarray(features, dtype=np.int16),
        np.asarray(multiplicity, dtype=np.int32),
    )
