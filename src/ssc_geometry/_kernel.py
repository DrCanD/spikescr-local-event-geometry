"""Scientific helpers retained from the recorded experiment implementation."""
from __future__ import annotations
import math
from typing import Any, Sequence
import numpy as np

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

def rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks

def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    rx, ry = rank_average(np.asarray(x, dtype=np.float64)), rank_average(np.asarray(y, dtype=np.float64))
    if float(rx.std()) == 0.0 or float(ry.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])

def gini_nonnegative(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    if len(x) == 0 or np.any(x < 0) or float(x.sum()) == 0.0:
        return 0.0
    x = np.sort(x)
    ranks = np.arange(1, len(x) + 1, dtype=np.float64)
    return float((2.0 * np.sum(ranks * x) / (len(x) * x.sum())) - (len(x) + 1) / len(x))

def source_bootstrap_mean_ci(
    values: np.ndarray, repetitions: int, seed: int
) -> tuple[float, list[float]]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(int(repetitions), len(values)))
    means = values[draws].mean(axis=1)
    return float(values.mean()), np.percentile(means, [2.5, 97.5]).astype(float).tolist()

def source_signflip_test(
    contrasts: np.ndarray, repetitions: int, seed: int
) -> dict[str, Any]:
    contrasts = np.asarray(contrasts, dtype=np.float64)
    contrasts = contrasts[np.isfinite(contrasts)]
    observed, ci = source_bootstrap_mean_ci(contrasts, repetitions, seed + 1)
    if len(contrasts) == 0:
        return {
            "n_sources": 0,
            "mean_contrast": None,
            "source_bootstrap95": [None, None],
            "two_sided_signflip_p": None,
        }
    rng = np.random.default_rng(seed)
    null = np.empty(int(repetitions), dtype=np.float64)
    for start in range(0, int(repetitions), 1000):
        stop = min(int(repetitions), start + 1000)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(stop - start, len(contrasts)))
        null[start:stop] = (signs * contrasts[None, :]).mean(axis=1)
    p = float((1 + np.sum(np.abs(null) >= abs(observed))) / (1 + len(null)))
    return {
        "n_sources": int(len(contrasts)),
        "mean_contrast": observed,
        "source_bootstrap95": ci,
        "two_sided_signflip_p": p,
        "repetitions": int(repetitions),
        "seed": int(seed),
    }

def benjamini_hochberg(p_values: Sequence[float | None]) -> list[float | None]:
    finite = [
        (index, float(value))
        for index, value in enumerate(p_values)
        if value is not None and math.isfinite(float(value))
    ]
    result: list[float | None] = [None] * len(p_values)
    if not finite:
        return result
    finite.sort(key=lambda item: item[1])
    count = len(finite)
    adjusted = np.empty(count, dtype=np.float64)
    running = 1.0
    for reverse_rank in range(count - 1, -1, -1):
        _, p_value = finite[reverse_rank]
        rank = reverse_rank + 1
        running = min(running, p_value * count / rank)
        adjusted[reverse_rank] = min(1.0, running)
    for (original_index, _), q_value in zip(finite, adjusted):
        result[original_index] = float(q_value)
    return result

def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if len(values) == 0 or float(weights.sum()) <= 0.0:
        return float("nan")
    return float(np.sum(values * weights) / np.sum(weights))

