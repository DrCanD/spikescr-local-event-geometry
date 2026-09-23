"""Source summaries from canonical stored outcomes.

Score margins are read from canonical records; other quantities are recomputed.
"""
from __future__ import annotations
from typing import Any
import numpy as np
from ._geometry_statistics import GEOMETRY_OPTIONS
def derive_transition_masks(
    label: int, clean_prediction: int, predictions: np.ndarray
) -> dict[str, np.ndarray]:
    predictions = np.asarray(predictions, dtype=np.int64)
    changed = predictions != int(clean_prediction)
    clean_correct = int(clean_prediction) == int(label)
    candidate_correct = predictions == int(label)
    adverse = changed & clean_correct & ~candidate_correct
    corrective = changed & (not clean_correct) & candidate_correct
    lateral = changed & ~(adverse | corrective)
    if not np.array_equal(changed, adverse | corrective | lateral):
        raise RuntimeError("Transition partition failed")
    code = np.zeros(len(predictions), dtype=np.uint8)
    code[adverse] = GEOMETRY_OPTIONS["transition_codes"]["adverse"]
    code[corrective] = GEOMETRY_OPTIONS["transition_codes"]["corrective"]
    code[lateral] = GEOMETRY_OPTIONS["transition_codes"]["lateral"]
    return {
        "changed": changed,
        "adverse": adverse,
        "corrective": corrective,
        "lateral": lateral,
        "preserved": ~changed,
        "code": code,
    }

def derive_source_geometry(
    corrected: dict[str, Any], arrays: dict[str, np.ndarray]
) -> dict[str, Any]:
    label = int(arrays["label"])
    clean_prediction = int(arrays["clean_prediction"])
    predictions = arrays["predictions"].astype(np.int64)
    from_t = arrays["from_t"].astype(np.int64)
    to_t = arrays["to_t"].astype(np.int64)
    feature = arrays["feature"].astype(np.int64)
    multiplicity = arrays["multiplicity"].astype(np.int64)
    masks = derive_transition_masks(label, clean_prediction, predictions)
    direction = to_t - from_t
    horizon = int(corrected["horizon_steps"])
    time_norm = from_t / max(1, horizon - 1)
    time_tertile = np.minimum((time_norm * 3).astype(np.int64), 2)
    feature_quartile = np.minimum((feature * 4 // 140).astype(np.int64), 3)

    row: dict[str, Any] = {
        "audit_rank": int(corrected["audit_rank"]),
        "source_index": int(corrected["source_index"]),
        "label": label,
        "clean_prediction": clean_prediction,
        "clean_correct": bool(clean_prediction == label),
        "clean_margin": float(corrected["clean_margin"]),
        "horizon_steps": horizon,
        "input_count": int(corrected["input_count"]),
        "unique_candidates": int(len(predictions)),
        "event_weighted_candidates": int(multiplicity.sum()),
        "class_change_rate": float(masks["changed"].mean()),
        "adverse_rate": float(masks["adverse"].mean()),
        "corrective_rate": float(masks["corrective"].mean()),
        "lateral_rate": float(masks["lateral"].mean()),
        "class_changed_unique": int(masks["changed"].sum()),
        "adverse_unique": int(masks["adverse"].sum()),
        "corrective_unique": int(masks["corrective"].sum()),
        "lateral_unique": int(masks["lateral"].sum()),
        "minimum_clean_class_margin": float(arrays["candidate_clean_class_margin"].min()),
        "minimum_true_class_margin": float(arrays["candidate_true_class_margin"].min()),
        "distinct_candidate_classes": int(len(np.unique(predictions))),
    }
    for value, name in ((-1, "previous"), (1, "next")):
        select = direction == value
        row[f"{name}_candidates"] = int(select.sum())
        row[f"{name}_class_change_rate"] = float(masks["changed"][select].mean()) if select.any() else float("nan")
        row[f"{name}_adverse_rate"] = float(masks["adverse"][select].mean()) if select.any() else float("nan")
    row["previous_minus_next_class_change_rate"] = (
        row["previous_class_change_rate"] - row["next_class_change_rate"]
    )
    for tertile in range(3):
        select = time_tertile == tertile
        row[f"time_tertile_{tertile + 1}_class_change_rate"] = (
            float(masks["changed"][select].mean()) if select.any() else float("nan")
        )
    for quartile in range(4):
        select = feature_quartile == quartile
        row[f"feature_quartile_{quartile + 1}_class_change_rate"] = (
            float(masks["changed"][select].mean()) if select.any() else float("nan")
        )

    lookup = {
        (int(t), int(f), int(d)): i
        for i, (t, f, d) in enumerate(zip(from_t, feature, direction))
    }
    pair_counts = np.zeros(4, dtype=np.int64)
    for (time_bin, channel, move_direction), previous_index in lookup.items():
        if move_direction != -1:
            continue
        next_index = lookup.get((time_bin, channel, 1))
        if next_index is None:
            continue
        previous_changed = int(masks["changed"][previous_index])
        next_changed = int(masks["changed"][next_index])
        pair_counts[2 * previous_changed + next_changed] += 1
    row.update(
        {
            "paired_neither_changed": int(pair_counts[0]),
            "paired_next_only_changed": int(pair_counts[1]),
            "paired_previous_only_changed": int(pair_counts[2]),
            "paired_both_changed": int(pair_counts[3]),
            "paired_direction_discordance": float(
                (pair_counts[2] - pair_counts[1]) / max(1, pair_counts.sum())
            ),
        }
    )
    return row

