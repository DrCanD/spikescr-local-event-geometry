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

The code verifies counts and candidate identities exactly. Statistical comparisons have their own explicitly recorded numerical policy. Do not interpret an intact ZIP, a passing subset or a successful file hash as a full model replay.
