# Reproduction protocol

## Distinguish the operations

`verify` reads file bytes and checks hashes. `reproduce` recalculates statistics from released arrays. `run_singleton_audit.py` runs the neural network. They answer different questions. Successful checksums alone do not validate a model or a statistical claim.

The local packaging tests exercise the recorded arrays and the orchestration helpers. They do not execute the full SpikeSCR model. Forward-kernel unit tests use an explicitly labelled small test double, while the real checkpoint is separately loaded with `weights_only=True` and its tensor-state digest is checked.

## Statistical regeneration

The tested analysis environment uses Python 3.13.5, NumPy 2.3.5 and SciPy 1.17.0. Its observed package versions are in `validation/analysis_environment.json`. This file is an observation, not a complete original-training lockfile.

The regenerated results are in `outputs/analysis_01/statistics` and `outputs/analysis_01/tables`. Publication-rounded Figure 4 CSVs are compared against the supplied figure data by the integration test. Full-precision tables remain available beside them. All figure data are derived from the arrays before comparison with the references.

A source utterance is the statistical sampling unit. The 725,070 candidate moves are not treated as independent utterances. The source-level bootstrap and permutation streams retain their recorded seeds, ordering and repetition counts. Trace means retain the original float32 reduction, while rates and resampling calculations use the recorded float64 operations. Slight reduction differences are listed in `numerical_agreement.json`; no reference value is substituted to force equality.

The tables contain classification metrics, not cross-entropy loss. `reproduce` does not claim to regenerate the stored benchmark loss, whose original batch aggregation differs from a single all-sample loss call.

## Validate the raw validation input

Obtain `ssc_valid.h5.gz` from the [official dataset page](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/) and decompress it. The recorded compressed MD5 is `555645b13e15fe270fbfe39cb763f805`. The uncompressed SHA-256 is stored in `configs/experiment.json`.

```bash
python -m pip install h5py
python -m ssc_geometry raw-validation --h5 /path/to/ssc_valid.h5
```

This command hashes the file before opening it as HDF5. It then reconstructs every panel input with two implementations and checks source labels. It does not search for or open the test set.

The terminal-bin convention is important. If the final event is exactly on a 5 ms boundary, it is included in the final existing bin according to the recorded implementation. Do not replace this transform with a different library default. A literal-loop test and the complete panel check cover this case.

## Configure inference deliberately

The recorded dependency probe reports `torch 2.11.0+cu128`. The three additional wheel versions and SHA-256 digests are in `configs/verified_wheels.json`. Install the matching PyTorch CUDA build in a separate environment according to [PyTorch's installation instructions](https://pytorch.org/get-started/previous-versions/). Do not let an unconstrained package update replace it.

One reproducible way to obtain the three pinned wheels is:

```bash
python -m pip download --only-binary=:all: --no-deps --require-hashes -r requirements-pinned-wheels.txt -d .cache/wheels
python -m pip install --no-deps --require-hashes -r requirements-pinned-wheels.txt --find-links .cache/wheels
```

Install the ordinary runtime dependencies required by SpikingJelly and HDF5, including a torchvision build compatible with the installed torch build, in that environment. The original versions of all those transitive packages are not recoverable from the stored dependency probe. Their absence is stated explicitly rather than replaced by a fabricated lockfile. Save the actual environment and use the conformance output, not version similarity alone, to judge a replay.

The loader validates the pinned source and configuration, applies the recorded 3D BatchNorm adapter, sets `use_ln=False`, selects the 5 ms input override and PyTorch neuron backend, and strictly loads the checkpoint. It also checks the trainable-parameter count and all seven named boundaries. Model weights, neuron equations and attention tensor reshapes are not changed.

## Preflight and full coverage

```bash
python scripts/prepare_upstream.py
python -m ssc_geometry checkpoint
python scripts/run_singleton_audit.py --mode preflight --out outputs/preflight_01
python scripts/run_singleton_audit.py --mode full --out outputs/singleton_01
```

Preflight sources are fixed before execution. Clean identity replacement and the full-output positive controls are checked. For the full run, each candidate prediction, its recorded score descriptors, six trace metrics at seven boundaries, and the applicable patched outputs are compared with the release arrays.

The release does not contain every unpatched 35-class score vector. Therefore candidate-score conformance uses the recorded descriptors and exact prediction, not an assertion of unavailable full-vector byte equality. Full patched vectors are available and checked. The source clean vectors are also checked in full.

A completed audit writes `conformance_report.json`. A report is complete only when it covers all 100 sources, 725,070 candidates and 25,820 seven-boundary replacement sets. Subsets, preflight probes, CPU diagnostics and cached outputs are reported separately. The numerical replay tolerances are stored in the experiment configuration and in each run manifest. A classification disagreement always fails, even if its score difference is small.

A completed source is stored atomically. Resume does not mix code, configuration, reference hashes or runtime environments. Files from an incomplete source may remain, but the source is recomputed rather than counted as finished.

## Replay the benchmark only when intended

```bash
python scripts/replay_benchmark.py --split validation --h5 /path/to/ssc_valid.h5 --out outputs/validation_benchmark_01
```

The benchmark program retains batch size 256, source order and padded-batch execution. The local-audit singleton path cannot replace it. Raw test access requires both an explicit test file and `--allow-official-test-replay`:

```bash
python scripts/replay_benchmark.py --split test --h5 /path/to/ssc_test.h5 --allow-official-test-replay --out outputs/test_benchmark_replication_01
```

This is a replication of a previously committed benchmark, not a new model-selection stage. The program never changes checkpoints or analysis settings. No raw test file was opened during repository preparation.

## What remains unverified

The packaging machine has no CUDA device and no MATLAB. It did not run the complete model replay, independent training, or MATLAB export. The upstream-download and GitHub-publication commands were not executed successfully in that environment. These are not converted into success claims by the unit tests. Keep the repository as a release candidate until its actual full-inference conformance report and author publication decisions are available.
