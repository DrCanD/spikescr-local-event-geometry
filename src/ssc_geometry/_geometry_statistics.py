from __future__ import annotations
from typing import Any
import numpy as np
from ._kernel import source_signflip_test, benjamini_hochberg, gini_nonnegative

GEOMETRY_OPTIONS={'time_bins': 20, 'feature_bins': 14, 'source_bootstrap_repetitions': 10000, 'source_signflip_repetitions': 10000, 'seed': 17062027, 'source_level_contrasts': ['previous-minus-next adjacent-bin class-change rate', 'late-minus-early time-tertile class-change rate', 'high-minus-low feature-quartile class-change rate'], 'multiple_testing': 'Benjamini-Hochberg FDR across the three source-level geometry contrasts', 'candidate_outputs': ['transition type and multiplicity', 'normalized time and feature coordinates', 'top-1, clean-class, and true-class margins', 'score L2, Linf, and cosine displacement from clean', '35-by-35 class-transition matrices', 'direction-by-time-by-feature transition maps'], 'transition_codes': {'class_preserved': 0, 'adverse': 1, 'corrective': 2, 'lateral': 3}}

def geometry_aggregates(
    candidate: dict[str, np.ndarray], source_rows: list[dict[str, Any]]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    time_bins = int(GEOMETRY_OPTIONS["time_bins"])
    feature_bins = int(GEOMETRY_OPTIONS["feature_bins"])
    time_id = np.minimum((candidate["time_normalized"] * time_bins).astype(np.int64), time_bins - 1)
    feature_id = np.minimum(
        (candidate["feature"].astype(np.int64) * feature_bins // 140), feature_bins - 1
    )
    direction_id = (candidate["direction"] > 0).astype(np.int64)
    transition = candidate["transition_code"].astype(np.int64)
    shape = (2, time_bins, feature_bins, 4)
    unique_grid = np.zeros(shape, dtype=np.int64)
    weighted_grid = np.zeros(shape, dtype=np.int64)
    np.add.at(unique_grid, (direction_id, time_id, feature_id, transition), 1)
    np.add.at(
        weighted_grid,
        (direction_id, time_id, feature_id, transition),
        candidate["multiplicity"].astype(np.int64),
    )
    transition_unique = np.zeros((35, 35), dtype=np.int64)
    transition_weighted = np.zeros((35, 35), dtype=np.int64)
    np.add.at(
        transition_unique,
        (candidate["clean_prediction"], candidate["candidate_prediction"]),
        1,
    )
    np.add.at(
        transition_weighted,
        (candidate["clean_prediction"], candidate["candidate_prediction"]),
        candidate["multiplicity"].astype(np.int64),
    )
    contrasts = np.asarray(
        [row["previous_minus_next_class_change_rate"] for row in source_rows],
        dtype=np.float64,
    )
    asymmetry = source_signflip_test(
        contrasts,
        int(GEOMETRY_OPTIONS["source_signflip_repetitions"]),
        int(GEOMETRY_OPTIONS["seed"]),
    )
    late_minus_early = source_signflip_test(
        np.asarray(
            [
                row["time_tertile_3_class_change_rate"]
                - row["time_tertile_1_class_change_rate"]
                for row in source_rows
            ],
            dtype=np.float64,
        ),
        int(GEOMETRY_OPTIONS["source_signflip_repetitions"]),
        int(GEOMETRY_OPTIONS["seed"]) + 10,
    )
    high_minus_low_feature = source_signflip_test(
        np.asarray(
            [
                row["feature_quartile_4_class_change_rate"]
                - row["feature_quartile_1_class_change_rate"]
                for row in source_rows
            ],
            dtype=np.float64,
        ),
        int(GEOMETRY_OPTIONS["source_signflip_repetitions"]),
        int(GEOMETRY_OPTIONS["seed"]) + 20,
    )
    geometry_tests = [asymmetry, late_minus_early, high_minus_low_feature]
    geometry_q = benjamini_hochberg(
        [test.get("two_sided_signflip_p") for test in geometry_tests]
    )
    for test, q_value in zip(geometry_tests, geometry_q):
        test["fdr_bh_q_across_geometry_contrasts"] = q_value
    adverse = np.asarray(
        [row["adverse_unique"] for row in source_rows if row["clean_correct"]],
        dtype=np.float64,
    )
    total_adverse = float(adverse.sum())
    shares = adverse / total_adverse if total_adverse else np.zeros_like(adverse)
    concentration = {
        "gini": gini_nonnegative(adverse),
        "herfindahl": float(np.sum(shares ** 2)),
        "effective_number_of_sources": float(1.0 / np.sum(shares ** 2)) if np.sum(shares ** 2) else 0.0,
    }
    aggregates = {
        "time_feature_direction_transition_unique": unique_grid,
        "time_feature_direction_transition_event_weighted": weighted_grid,
        "class_transition_unique": transition_unique,
        "class_transition_event_weighted": transition_weighted,
        "time_bin_edges_normalized": np.linspace(0.0, 1.0, time_bins + 1, dtype=np.float32),
        "feature_bin_edges": np.arange(feature_bins + 1, dtype=np.int16) * (140 // feature_bins),
    }
    summary = {
        "directional_asymmetry_previous_minus_next": asymmetry,
        "temporal_gradient_late_minus_early_tertile": late_minus_early,
        "feature_gradient_high_minus_low_quartile": high_minus_low_feature,
        "adverse_source_concentration": concentration,
    }
    return aggregates, summary

