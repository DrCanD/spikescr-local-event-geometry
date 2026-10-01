# Data files

Every file under `data/`, with its array keys or columns and the manuscript element it serves. Source indices,
class labels and candidate indices are zero-based. A source index is the row of the utterance in the official SSC
validation split. `audit_rank` is the fixed panel order (0–99), not a performance rank.

Scores everywhere are the public readout: the sum over time steps of the softmax of the per-step logits; the
prediction is the argmax (first index on ties). The binary files (`.npz`, `.pt`) are the files produced by the
experiment runs; only their names were chosen for this repository (`provenance/records.json` lists the original
names and hashes).

## `data/model/` — the frozen checkpoint (§3.2)

| File | Content |
| --- | --- |
| `checkpoint.pt` | PyTorch checkpoint of the audited SpikeSCR model: `model_state_dict` (3,302,416 parameters) and `epoch` (281, zero-based; epoch 282 in the manuscript). Loaded with `weights_only=True`. Hashes: `configs/experiment.json` → `checkpoint_sha256`, `state_dict_sha256`. |

## `data/panel/` — the audit panel (§3.3)

| File | Keys / columns |
| --- | --- |
| `inputs.npz` | `source_XXXXX` → `[horizon, 140]` uint16 count tensor of each panel source (5 ms bins, 5-channel units, horizon = bins up to the last event; not padded). |
| `sources.csv` | `audit_rank`, `source_index`, `label`, `horizon_steps`, `input_count` (total spike count), `unique_candidates` (neighbors of the source), `input_array_sha256` (SHA-256 of the little-endian uint16 C-order bytes). |
| `clean_scores.npz` | `source_index` `[100]`, `scores` `[100, 35]` float32: the clean native-singleton score vectors (the canonical audit path). |

## `data/benchmark/` — Table 2

| File | Keys |
| --- | --- |
| `validation_batch256.npz` | `source_indices`, `labels`, `predictions` (all `[9981]`), `confusion` `[35, 35]`: the batch-256 validation evaluation. |
| `test_batch256.npz` | `source_indices`, `labels`, `predictions`, `scores` `[20382, 35]`, `lengths`, `confusion`: the single batch-256 evaluation of the official test split. No command opens the raw test file. |

## `data/neighborhood/` — the complete census (§4.2; Tables 3, 7; Fig. 3)

`candidates.npz`: 725,070 rows in panel order, then candidate order within each source. At an occupied cell the
move to the preceding bin is enumerated before the move to the following bin.

| Key | Meaning |
| --- | --- |
| `audit_rank`, `source_index`, `candidate_index` | identity (candidate index is zero-based within its source) |
| `label`, `clean_prediction`, `candidate_prediction` | true class, clean prediction of the source, prediction for the candidate |
| `from_bin`, `to_bin`, `feature`, `direction` | the move (one count from `from_bin` to `to_bin` in `feature`; `direction` = `to_bin − from_bin` ∈ {−1, +1}) |
| `multiplicity` | spike count at the origin cell (event-weighted denominator) |
| `transition_code` | 0 class-preserved, 1 adverse, 2 corrective, 3 lateral |
| `time_normalized`, `feature_normalized` | `from_bin / (horizon − 1)`, `feature / 139` |
| `candidate_top1_margin`, `candidate_clean_class_margin`, `candidate_true_class_margin` | margins of the candidate score vector (top-two; clean class minus best other; true class minus best other) |
| `delta_clean_class_score`, `delta_true_class_score`, `score_l2_change`, `score_linf_change`, `score_cosine_distance` | displacement of the candidate score vector from the clean score vector |

Full 35-class score vectors of unpatched candidates are not included; the descriptors above are.

`sources.csv`: one row per source (100), all recomputed by `census` from `candidates.npz`: `audit_rank`,
`source_index`, `label`, `clean_prediction`, `clean_correct`, `clean_margin`, `horizon_steps`, `input_count`,
`unique_candidates`, `event_weighted_candidates`, `class_change_rate`, `adverse_rate`, `corrective_rate`,
`lateral_rate`, `class_changed_unique`, `adverse_unique`, `corrective_unique`, `lateral_unique`,
`minimum_clean_class_margin`, `minimum_true_class_margin`, `distinct_candidate_classes`, `previous_*` / `next_*`
(moves to the preceding / following bin), `previous_minus_next_class_change_rate`, `time_tertile_{1,2,3}_class_change_rate`,
`feature_quartile_{1..4}_class_change_rate`, `paired_*` (preceding/following move pairs at the same cell).

## `data/internal/` — activation change and replacement (§4.4–4.6; Tables 5, 6, 8; Fig. 4)

`activation_metrics_and_replacements.npz`:

| Key | Shape | Meaning |
| --- | --- | --- |
| `source_index`, `candidate_index`, `transition_code`, `multiplicity` | `[725070]` | aligned with `candidates.npz` |
| `stage_names` | `[7]` | `stem, attention_1, local_1, block_1, attention_2, local_2, block_2` |
| `metric_names` | `[6]` | `mean_abs_change, rms_change, relative_l2_change, cosine_distance, activity_flip_rate, mean_signed_change` |
| `trace_metrics` | `[725070, 7, 6]` float32 | activation change of the candidate relative to the clean activation at each boundary; relative L2 uses `max(‖clean‖₂, 1e-12)`; activity flip compares the predicates `activation > 0.5` |
| `trace_provenance` | `[725070]` | 1 = reused singleton trace, 2 = recomputed singleton trace (51,560 / 673,510); an execution-history field |
| `patch_source_index`, `patch_candidate_index` | `[25820]` | the class-changing candidates (rows with `transition_code ≠ 0`, in census order) |
| `patch_scores` | `[25820, 7, 35]` float32 | score vector after replacing the candidate's activation at one boundary by the clean activation |
| `patch_prediction`, `patch_clean_class_margin`, `patch_true_class_margin` | `[25820, 7]` | derived from `patch_scores` |
| `contract_sha256` | scalar | hash of the contract of the run that produced the file |

Replacement at the stem, block 1 or block 2 reproduces the clean score vector exactly (positive controls).

## `data/search/` — search policies (§4.3; Table 4; Supplementary S2)

| File | Content |
| --- | --- |
| `margin_policies.json` | the allocation policies (uniform; lowest 10/25/50% by clean margin; 50/50 uniform + quartile), budgets, tie breaks and the cutoff-sweep specification |
| `candidate_scores.npz` | `source_index` `[100]`, `source_offsets` `[101]` (candidate slice of each source, ascending source index), `policy_names` (`runner_up`, `max_directional_gain`, `max_predicted_target_margin`), `move_scores` `[3, 725070]` float32 |
| `candidate_ranks.npz` | the same identity keys, `ranked_candidate_index` `[3, 725070]` (within-source candidate indices in descending move-score order, candidate-index tie break), `surrogate_clean_scores` `[100, 35]`, `rival_class` `[100]` |
| `runner_up_gradients.npz` | `runner_up_input_gradients` (ragged float32, one `[horizon, 140]` block per source delimited by `gradient_offsets` `[101]`), `horizon` `[100]`; the runner-up move score is `g[to, feature] − g[from, feature]` |
| `cutoff_sensitivity.csv` | the S2 sweep: `ranking` (raw margin / margin per horizon), `threshold_percent`, `selected_source_count`, `total_candidate_queries`, `expected_detected_sources`, `probability_detect_all13`, `source1627_*` |
| `witnesses.json` | the first adverse candidate of each vulnerable source under each ranking, re-executed as a fresh forward pass and compared with the canonical prediction |
| `summary.json` | discovery counts of the all-target run as written by that run |

The full 34-target gradient tensors are not included; their move scores and rankings are. `scripts/run_search_gradients.py`
recomputes both.

## `data/replicas/` — count-readout replicas (§4.7; Table 9)

| File | Content |
| --- | --- |
| `source_outcomes.csv` | 200 rows (100 sources × replicas A/B): `seed`, `source_index`, `label`, `clean_prediction`, `clean_correct`, `clean_top2_margin`, `complete`, `candidates`, `candidates_done`, `T0`, `T1`, `T2`, `T3`, `t3_rate`, `adverse`, `corrective`, `lateral` |
| `paired_sources.csv` | the same outcomes paired by source: `A_*` and `B_*` counts and `*_source_rate` (count / neighbors) |
| `readout_partition.csv` | per replica and scope: `trace_changed_neighbors`, `final_trace_changed_counts_preserved`, `earlier_trace_changed_final_trace_preserved` and their fractions (the final-layer row of Table 9) |
| `validation_A.npz`, `validation_B.npz` | `source_indices`, `labels`, `predictions`, `logits` `[9981, 35]`: full-validation outputs of each replica (7,214 and 7,239 correct) |
| `audit_contract.json` | the contract of the replica audit: analysis settings and seeds, both checkpoints' hashes and configuration, dataset hash, environment, panel selection, protocol |
| `protocol.json` | the decision rules fixed before the audit |
| `summary.json` | the audit's own summary: integrity gates, per-replica results, decisions |
| `cohorts.json` | the paired cohorts (full panel, both correct, A-wrong/B-correct, A-correct/B-wrong, both wrong) and the bootstrap settings (20,000 repetitions, seed 20260929) |

The outcomes: T0 trace unchanged; T1 trace changed, counts preserved; T2 counts changed, class preserved;
T3 class changed. `historical_correct` in the contract and summary is the training-time validation count of each
replica (7,213 / 7,241); the replay in `validation_*.npz` gives 7,214 / 7,239, and the difference is reported, not
corrected. The replicas' native neighborhood uses 10 ms bins; their 1,297,705 neighbors per replica are never
pooled with the 725,070-neighbor SpikeSCR census. Replica weights and raw candidate maps are not included.

## `data/execution/` — execution controls (§4.1, 4.8; Appendix A)

All full-validation arrays have 9,981 rows in ascending source order.

| File | Keys |
| --- | --- |
| `validation_gpu_conditions.npz` | `source_index`, `label`, `native_horizon`, `padded_horizon` (maximum horizon of the source's batch-256 group), `input_sha256` (hash of the float32 input), four `[9981, 35]` score matrices `B1_native_scores`, `B1_padding_matched_scores`, `B256_scores`, `B256_reversed_scores` and their `*_prediction`, `CPU_B1_prediction`, `archived_GPU_B256_prediction`, `replayed_source_index` (10 identical-input replays), `contract_sha256` |
| `validation_cpu_singleton.npz` | `source_index`, `label`, `prediction`, `scores`, `horizon`, `input_sha256`, `confusion`, `archived_B256_prediction`, `B1_B256_prediction_disagreement`, `contract_sha256` |
| `qk_isolation.npz` | `source_index`, `label`, `original_scores`, `original_reversed_scores`, `qk_isolated_scores`, `qk_isolated_reversed_scores`, `contract_sha256` |
| `candidate_replay.npz` | the 25,820 class-changing candidates: `geometry_row_index` (row in `candidates.npz`), `source_index`, `candidate_index`, `label`, `from_bin`, `to_bin`, `feature`, `canonical_candidate_prediction`, `canonical_transition_code`, `CPU_scores`, `GPU_scores` `[25820, 35]`, `CPU_prediction`, `GPU_prediction`, `CPU_clean_prediction`, `GPU_clean_prediction`, `CPU_transition_code`, `GPU_transition_code`, `contract_sha256` |
| `summary.json` | the summaries written by the two control runs (`gpu_conditions`, `isolation_and_replay`), including the padding decomposition (408 sources) and the three pair controls whose per-source arrays are not included |

The `contract_sha256` values are the hashes of `provenance/contracts/execution_gpu_conditions.json` (GPU conditions),
`provenance/contracts/gradient_runner_up.json` (CPU singleton run) and `provenance/contracts/execution_isolation_and_replay.json`
(isolation and candidate replay); `execution` checks them.

## `data/reference/` — comparison targets

| File | Content |
| --- | --- |
| `expected_values.json` | every number printed in the manuscript: `id`, `location` (table or section), `path` into the regenerated results, `value`, printed `decimals`, `scale`; `python -m ssc_geometry paper` writes the comparison to `report.json` |
| `benchmark.json`, `neighborhood.json`, `geometry.json`, `internal.json` | regenerated census statistics (classification metrics, outcome totals, source-level geometry contrasts, internal contrasts and replacement rates) |
| `search.json`, `replicas.json`, `execution.json` | regenerated search, replica and execution-control statistics |

The commands compare every leaf of their results with these files (counts and predictions exactly; p and q values
and float64 rates to 1e-12; float32-derived means and intervals to 1e-6).

## `configs/`

| File | Content |
| --- | --- |
| `experiment.json` | checkpoint and panel settings, transform parameters, boundary names and module paths, metric names, statistical seeds and repetition counts, tolerances, the validation and test HDF5 hashes |
| `upstream.json` | the eight SpikeSCR source files and their git blob hashes at commit `095f418` |
| `environment.json`, `verified_wheels.json` | the reference inference environment (torch 2.11.0+cu128) and the pinned wheel hashes |

## `provenance/`

| File | Content |
| --- | --- |
| `manifest.json`, `SHA256SUMS.txt` | SHA-256 of every tracked file (`python -m ssc_geometry verify`; regenerate with `scripts/update_manifest.py`) |
| `contracts/*.json` | the contracts of the four model runs (`execution_gpu_conditions`, `execution_isolation_and_replay`, `gradient_runner_up`, `gradient_all_target`), kept verbatim because their hash is stored inside the output arrays |
| `records.json` | the file history of this repository: original name → current name and hash of every renamed or rewritten record, the removed packaging files, the executed-script hashes and the scripts that supersede them |
| `validation/*.json` | the analysis environment, the numerical-agreement record, the CPU portability probe (source 4959), the CPU model preflight and the upstream-source check |
