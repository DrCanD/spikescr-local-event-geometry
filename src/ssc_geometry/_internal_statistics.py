"""Recorded statistical aggregation, with descriptive interface names.

The numerical aggregation body and random-number streams are unchanged.
Reference artifacts are not read by this module.
"""
from __future__ import annotations
import math
from typing import Any
import numpy as np
from ._kernel import weighted_mean, source_signflip_test, source_bootstrap_mean_ci, benjamini_hochberg
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
