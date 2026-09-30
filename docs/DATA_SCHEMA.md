# Data schema

Source IDs, class labels and candidate indices are zero-based. A source ID refers to its row in the official SSC validation split. Audit rank is the fixed panel order, not a performance rank.

## Panel

`inputs.npz` contains `source_XXXXX` arrays of shape `[horizon,140]` and dtype uint16. Horizons are source-specific; they are not all padded to 200. `clean_scores.npz` contains a 100-element `source_index` array and a `[100,35]` float32 `scores` array. The manifest records source labels and SHA-256 digests of little-endian uint16 C-order array bytes.

The clean score is the sum, across time, of the softmax probabilities of the time-step logits. It is not the softmax of summed logits. The prediction is its argmax, with the same first-index tie behavior as the original implementation.

## Neighborhood

Every field in `candidate_records.npz` has 725,070 rows in audit-rank and within-source candidate order. At an occupied cell, the preceding-bin move is enumerated before the following-bin move when each is valid.

`from_bin`, `to_bin` and `feature` specify the unit-count transfer. `multiplicity` is the count at the occupied source cell, not the number of destination events. `direction` is -1 or +1. All labels and predictions use the 35-class zero-based convention.

The `transition_code` values are 0 for class-preserved, 1 for adverse, 2 for corrective, and 3 for lateral. A lateral transition starts from a wrong clean prediction and ends at a different wrong class. Corrective and lateral outcomes must not be counted as adverse outcomes.

The stored score descriptors include the top-two margin, clean-class margin, true-class margin, clean and true score changes, L2 displacement, maximum absolute displacement, and cosine distance. Full unpatched score vectors are not present in this archive.

## Internal metrics and replacement

The internal archive has `trace_metrics` with shape `[725070,7,6]`, dtype float32. The stage order is `stem`, `attention_1`, `local_1`, `block_1`, `attention_2`, `local_2`, `block_2`. The metric order is mean absolute change, root-mean-square change, relative L2 change, cosine distance, activity-state change, and mean signed change. `stage_names` and `metric_names` store these orders explicitly.

Relative L2 uses max(clean L2 norm, 1e-12) in the denominator. Activity-state change compares the predicates activation > 0.5, not merely nonzero activation.

`patch_scores` has shape `[25820,7,35]`, while `patch_prediction`, `patch_clean_class_margin` and `patch_true_class_margin` each have shape `[25820,7]`. Replacement records are aligned through `patch_source_index` and `patch_candidate_index`, not by assuming their row numbers equal neighborhood row numbers.

`trace_provenance` uses code 1 for eligible reused singleton traces and code 2 for recomputed singleton traces. Their counts are 51,560 and 673,510, respectively. It is an execution-history field, not a scientific outcome. `contract_sha256` retains the original logical contract hash; descriptive public filenames do not create a new historical contract.

For class-changing candidates, complete-stem and complete-block replacement restores the clean score vector. These are positive controls. They do not establish unique causality of a branch. For an initially wrong source, restoring the clean prediction restores an error; it is not ground-truth repair.

## Benchmarks

The validation and test archives retain the original ordered source indices, labels, predictions and confusion matrices. The test archive also contains its `[20382,35]` scores and transformed sequence lengths. These are recorded benchmark outputs, not raw test events.

## Reproducibility boundaries

The code verifies counts and candidate identities exactly. Statistical comparisons have their own explicitly recorded numerical policy. A passing subset or file hash does not establish a full model replay.

## Search policies

`data/search_policies/candidate_rankings.npz` concatenates all 100 sources in
ascending source-ID order. `source_offsets` has 101 entries delimiting each
source's candidate slice. Within a slice, candidate IDs are zero-based and
follow the original enumeration. `ranked_candidate_index` and `move_scores`
(in `candidate_policy_scores.npz`) have shape `[3,725070]`. Their source IDs,
offsets and policy names must agree. Policy order is `runner_up`, `max_directional_gain`,
`max_predicted_target_margin`. A rank contains within-source candidate IDs,
not row numbers in the original globally ordered census.

`runner_up_input_gradients.npz` stores `runner_up_input_gradients`, a ragged float32 sequence delimited by
`gradient_offsets`; reshape each source to `[horizon,140]`. Its directional
score is gradient at the destination minus gradient at the origin.
`surrogate_clean_scores` and `rival_class` preserve the scores and target used.
The input-gradient values, policy scores and ranks retain their original dtypes.
Full 34-target gradient tensors and per-target directional gains were checked
during compaction but are not duplicated here. The original per-source hashes,
archive hash, target-max checks and runner-up parity are in
`compaction_provenance.json`. Contracts and first-witness checks are included.

## Count-readout replicas

`data/replicas/source_results.csv` contains 100 source records for each of seeds
341969035 (A) and 554720563 (B), with counts T0–T3 and adverse/corrective/lateral
T3 directions. `paired_source_data.csv` aligns those records by source ID.
These replicas use native 10 ms bins and 1,297,705 neighbors each; their
denominators must not be pooled with the 725,070-neighbor SpikeSCR census.
Their readout uses global and four-window integer final-layer spike counts.
The source summaries, original contract and preregistration are preserved;
raw replica hidden tensors and their complete candidate maps are not bundled.

`validation_replica_A.npz` and `validation_replica_B.npz` each store
`source_indices`, `labels`, `predictions`, and `[9981,35]` float32 `logits`.
The recorded replay yields 7,214 and 7,239 correct labels. Original historical
counts of 7,213 and 7,241 remain reported in the recorded summary; they were not
silently substituted for the fresh replay.

## Execution controls

All full-validation records have 9,981 rows in ascending source-ID order.

- `gpu_validation_predictions.npz` contains labels, native/padded horizons,
  transformed-input hashes, and four `[9981,35]` float32 score matrices: native
  singleton, padding-matched singleton, batch 256, and reversed batch 256.
- `cpu_validation_predictions.npz` contains the separately recorded native CPU
  singleton scores and predictions, identical input hashes and native horizons.
- `qk_isolation_validation.npz` contains original and source-isolated batch score
  matrices under both orders, all `[9981,35]` float32.
- `paired_candidate_predictions.npz` contains the 25,820 original class-changing
  candidates' canonical row IDs, coordinates, labels and transition codes, both
  devices' `[25820,35]` float32 scores, and paired clean/candidate predictions.
  Its `geometry_row_index` matches exactly the nonzero transition rows of
  `candidate_records.npz`. Originally preserved candidates are not in this file.

Original run contracts and summaries retain their logical hashes and environment
records. Files were renamed descriptively without rewriting their bytes.
Summaries additionally describe 408 padding-prefix diagnostic sources and three
pair controls; their full diagnostic arrays are not included here. The
array-only command reports only what it recomputes from the bundled arrays.
