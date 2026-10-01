"""Statistical procedures of the audit: rank statistics, source-level bootstrap and sign-flip tests,
false-discovery-rate control, candidate-geometry aggregates and internal-activation contrasts.

The random-number streams and aggregation bodies are those of the original analysis."""
from __future__ import annotations
import math
from typing import Any, Sequence
import numpy as np

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


# ---------------------------------------------------------------- candidate geometry aggregates
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


# ---------------------------------------------------------------- internal activation statistics

OPTIONS = {"primary_internal_metric": "relative_l2_change", "independent_unit": 'source utterance; candidate moves are nested', "statistical_seed": 19062027, "source_signflip_repetitions": 10000, "source_bootstrap_repetitions": 10000, "positive_control_stages": ["stem", "block_1", "block_2"]}

def recompute_internal(geometry, archive, clean_scores):
    source_index = archive["source_index"]
    candidate_index = archive["candidate_index"]
    transition_code = archive["transition_code"]
    multiplicity = archive["multiplicity"]
    trace_metrics = archive["trace_metrics"]
    patch_source_index = archive["patch_source_index"]
    patch_candidate_index = archive["patch_candidate_index"]
    patch_scores = archive["patch_scores"]
    patch_prediction = archive["patch_prediction"]
    patch_clean_margin = archive["patch_clean_class_margin"]
    patch_true_margin = archive["patch_true_class_margin"]
    stage_names = archive["stage_names"].tolist()
    metric_names = archive["metric_names"].tolist()
    changed = transition_code != 0
    patch_clean_scores = np.stack([clean_scores[int(s)] for s in patch_source_index])
    group_names = ["class_preserved", "adverse", "corrective", "lateral"]
    trace_group_means_unique: dict[str, Any] = {}
    trace_group_means_event_weighted: dict[str, Any] = {}
    for code, group_name in enumerate(group_names):
        selected = transition_code == code
        trace_group_means_unique[group_name] = {}
        trace_group_means_event_weighted[group_name] = {}
        for stage_index, stage_name in enumerate(stage_names):
            trace_group_means_unique[group_name][stage_name] = {}
            trace_group_means_event_weighted[group_name][stage_name] = {}
            for metric_index, metric_name in enumerate(metric_names):
                values = trace_metrics[selected, stage_index, metric_index]
                trace_group_means_unique[group_name][stage_name][metric_name] = (
                    float(values.mean()) if len(values) else None
                )
                weighted = weighted_mean(values, multiplicity[selected])
                trace_group_means_event_weighted[group_name][stage_name][metric_name] = (
                    weighted if math.isfinite(weighted) else None
                )

    patch_transition = transition_code[changed]
    patch_multiplicity = multiplicity[changed]
    patch_labels = geometry["label"][changed].astype(np.int64)
    patch_clean_prediction = geometry["clean_prediction"][changed].astype(np.int64)
    patch_summary: dict[str, Any] = {}
    for code, group_name in enumerate(group_names[1:], start=1):
        selected = patch_transition == code
        patch_summary[group_name] = {}
        for stage_index, stage_name in enumerate(stage_names):
            predictions = patch_prediction[selected, stage_index].astype(np.int64)
            weights = patch_multiplicity[selected]
            restored_clean = predictions == patch_clean_prediction[selected]
            restored_true = predictions == patch_labels[selected]
            positive_clean_margin = patch_clean_margin[selected, stage_index] > 0
            patch_summary[group_name][stage_name] = {
                "n_unique": int(selected.sum()),
                "n_event_weighted": int(weights.sum()),
                "restored_clean_prediction_rate_unique": float(
                    restored_clean.mean()
                ) if len(predictions) else None,
                "restored_clean_prediction_rate_event_weighted": weighted_mean(
                    restored_clean.astype(float), weights
                ) if len(predictions) else None,
                "restored_ground_truth_rate_unique": float(
                    restored_true.mean()
                ) if len(predictions) else None,
                "restored_ground_truth_rate_event_weighted": weighted_mean(
                    restored_true.astype(float), weights
                ) if len(predictions) else None,
                "positive_clean_class_margin_rate_unique": float(
                    positive_clean_margin.mean()
                ) if len(predictions) else None,
            }

    primary_index = metric_names.index(str(OPTIONS["primary_internal_metric"]))
    unique_sources = np.unique(source_index)
    inference: dict[str, Any] = {
        "independent_unit": OPTIONS["independent_unit"],
        "primary_metric": OPTIONS["primary_internal_metric"],
        "transition_vs_preserved": {},
        "patch_restoration": {},
        "global_vs_local": {},
    }
    base_seed = int(OPTIONS["statistical_seed"])
    repetitions = int(OPTIONS["source_signflip_repetitions"])
    bootstrap_repetitions = int(OPTIONS["source_bootstrap_repetitions"])
    for code, group_name in enumerate(group_names[1:], start=1):
        inference["transition_vs_preserved"][group_name] = {}
        inference["patch_restoration"][group_name] = {}
        for stage_index, stage_name in enumerate(stage_names):
            contrasts: list[float] = []
            restore_clean_rates: list[float] = []
            restore_true_rates: list[float] = []
            for source in unique_sources:
                in_source = source_index == source
                selected = in_source & (transition_code == code)
                preserved = in_source & (transition_code == 0)
                if selected.any() and preserved.any():
                    contrasts.append(
                        float(
                            trace_metrics[selected, stage_index, primary_index].mean()
                            - trace_metrics[preserved, stage_index, primary_index].mean()
                        )
                    )
                patch_selected = (patch_source_index == source) & (
                    patch_transition == code
                )
                if patch_selected.any():
                    predictions = patch_prediction[
                        patch_selected, stage_index
                    ].astype(np.int64)
                    restore_clean_rates.append(
                        float(
                            np.mean(
                                predictions
                                == patch_clean_prediction[patch_selected]
                            )
                        )
                    )
                    restore_true_rates.append(
                        float(np.mean(predictions == patch_labels[patch_selected]))
                    )
            inference["transition_vs_preserved"][group_name][stage_name] = (
                source_signflip_test(
                    np.asarray(contrasts, dtype=np.float64),
                    repetitions,
                    base_seed + 1000 + 10 * code + stage_index,
                )
            )
            clean_mean, clean_ci = source_bootstrap_mean_ci(
                np.asarray(restore_clean_rates, dtype=np.float64),
                bootstrap_repetitions,
                base_seed + 2000 + 10 * code + stage_index,
            )
            true_mean, true_ci = source_bootstrap_mean_ci(
                np.asarray(restore_true_rates, dtype=np.float64),
                bootstrap_repetitions,
                base_seed + 3000 + 10 * code + stage_index,
            )
            inference["patch_restoration"][group_name][stage_name] = {
                "n_sources": int(len(restore_clean_rates)),
                "mean_restored_clean_prediction_rate": clean_mean,
                "source_bootstrap95_restored_clean_prediction_rate": clean_ci,
                "mean_restored_ground_truth_rate": true_mean,
                "source_bootstrap95_restored_ground_truth_rate": true_ci,
            }

    for code, group_name in enumerate(group_names):
        inference["global_vs_local"][group_name] = {}
        for block_index in (1, 2):
            attention_index = stage_names.index(f"attention_{block_index}")
            local_index = stage_names.index(f"local_{block_index}")
            trace_contrasts: list[float] = []
            patch_contrasts: list[float] = []
            for source in unique_sources:
                selected = (source_index == source) & (transition_code == code)
                if selected.any():
                    trace_contrasts.append(
                        float(
                            trace_metrics[
                                selected, attention_index, primary_index
                            ].mean()
                            - trace_metrics[
                                selected, local_index, primary_index
                            ].mean()
                        )
                    )
                if code != 0:
                    patch_selected = (patch_source_index == source) & (
                        patch_transition == code
                    )
                    if patch_selected.any():
                        target = patch_clean_prediction[patch_selected]
                        attention_restored = np.mean(
                            patch_prediction[
                                patch_selected, attention_index
                            ].astype(np.int64)
                            == target
                        )
                        local_restored = np.mean(
                            patch_prediction[
                                patch_selected, local_index
                            ].astype(np.int64)
                            == target
                        )
                        patch_contrasts.append(
                            float(attention_restored - local_restored)
                        )
            block_result: dict[str, Any] = {
                "attention_minus_local_trace_divergence": source_signflip_test(
                    np.asarray(trace_contrasts, dtype=np.float64),
                    repetitions,
                    base_seed + 4000 + 100 * code + block_index,
                )
            }
            if code != 0:
                block_result[
                    "attention_minus_local_clean_prediction_restoration"
                ] = source_signflip_test(
                    np.asarray(patch_contrasts, dtype=np.float64),
                    repetitions,
                    base_seed + 5000 + 100 * code + block_index,
                )
            inference["global_vs_local"][group_name][
                f"block_{block_index}"
            ] = block_result

    transition_family: list[dict[str, Any]] = []
    for group_name in group_names[1:]:
        transition_family.extend(
            inference["transition_vs_preserved"][group_name][stage_name]
            for stage_name in stage_names
        )
    transition_q = benjamini_hochberg(
        [item.get("two_sided_signflip_p") for item in transition_family]
    )
    for item, q_value in zip(transition_family, transition_q):
        item["fdr_bh_q_within_transition_vs_preserved_family"] = q_value
    global_trace_family: list[dict[str, Any]] = []
    global_patch_family: list[dict[str, Any]] = []
    for group_name in group_names:
        for block_index in (1, 2):
            block = inference["global_vs_local"][group_name][f"block_{block_index}"]
            global_trace_family.append(block["attention_minus_local_trace_divergence"])
            if group_name != "class_preserved":
                global_patch_family.append(
                    block["attention_minus_local_clean_prediction_restoration"]
                )
    trace_q = benjamini_hochberg(
        [item.get("two_sided_signflip_p") for item in global_trace_family]
    )
    for item, q_value in zip(global_trace_family, trace_q):
        item["fdr_bh_q_within_global_vs_local_trace_family"] = q_value
    patch_q = benjamini_hochberg(
        [item.get("two_sided_signflip_p") for item in global_patch_family]
    )
    for item, q_value in zip(global_patch_family, patch_q):
        item["fdr_bh_q_within_global_vs_local_patch_family"] = q_value

    positive_control: dict[str, Any] = {}
    for stage_name in OPTIONS["positive_control_stages"]:
        stage_index = stage_names.index(stage_name)
        difference = np.max(
            np.abs(patch_scores[:, stage_index] - patch_clean_scores), axis=1
        )
        positive_control[stage_name] = {
            "max_abs_to_clean_scores": float(difference.max()),
            "prediction_restoration_rate": float(
                np.mean(
                    patch_prediction[:, stage_index].astype(np.int64)
                    == patch_clean_prediction
                )
            ),
        }

    return {"trace_group_means_unique": trace_group_means_unique,
            "trace_group_means_event_weighted": trace_group_means_event_weighted,
            "clean_activation_patch": patch_summary,
            "source_level_inference": inference, "positive_controls": positive_control}
