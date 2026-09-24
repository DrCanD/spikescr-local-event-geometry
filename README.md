# SpikeSCR Local Event Geometry

[![Validation](https://github.com/DrCanD/spikescr-local-event-geometry/actions/workflows/validate.yml/badge.svg?branch=main)](https://github.com/DrCanD/spikescr-local-event-geometry/actions/workflows/validate.yml)

Research code and data accompanying **Stable predictions can hide substantial temporal changes in a high-accuracy spiking speech classifier** by **İsmail Can Dikmen**.

[Overview](#overview) · [Quick start](#quick-start) · [Model replay](#model-replay) · [Repository contents](#repository-contents) · [Validation](#validation) · [Citation](#citation)

## Overview

Can a spiking speech classifier preserve its predicted word while its internal activity changes substantially?

This study examines that question using a frozen [SpikeSCR](https://github.com/JackieWang9811/SpikeSCR) checkpoint on Spiking Speech Commands. For a fixed panel of 100 validation utterances, it exhaustively evaluates **725,070 neighboring inputs**, measures activation changes at **seven network boundaries**, and tests whether replacing a perturbed activation with its clean counterpart restores the original decision.

Each neighboring input moves one integrated count to the previous or next **5 ms time bin** within the same feature. The transformation preserves the input horizon and total count. The findings apply to this defined neighborhood and panel.

The repository supports two workflows:

| Workflow | Purpose | Requirements |
| --- | --- | --- |
| **Recalculate the reported statistics** | Recompute numerical tables from the included predictions, activation metrics and intervention outputs. | Analysis dependencies; no GPU or raw dataset required. |
| **Run the model again** | Generate new predictions, activation metrics and intervention outputs, then compare them with the released records. | Frozen checkpoint, verified upstream source and the documented inference environment. |

The reproduction target is the technical experiment at the supplied checkpoint. The [training record](docs/TRAINING.md) documents how that checkpoint was obtained; independent retraining and manuscript presentation assets are outside this package's scope.

## Experiment at a glance

| Item | Recorded value |
| --- | --- |
| Dataset and input representation | Spiking Speech Commands; 35 classes; 700 channels integrated into 140 features |
| Checkpoint | Training seed 312; selected epoch 282 |
| Parameters | 3,302,416 total; 3,302,400 trainable; 16 fixed rotary-frequency parameters |
| Validation / test benchmark | 8,617 / 9,981 correct; 17,247 / 20,382 correct |
| Local audit | 100 validation utterances; 725,070 unique neighbors |
| Preserved / adverse / corrective / lateral outcomes | 699,250 / 5,157 / 9,031 / 11,632 |
| Event-weighted candidate count | 1,659,158 |
| Initially correct / adverse-sensitive sources | 84 / 13 |
| Internal measurements | Six activation-change metrics at seven boundaries |
| Activation replacement | 25,820 class-changing candidates × seven boundaries = 180,740 patched score vectors |

**Preserved** means the predicted class stays the same. An **adverse** transition changes a correct prediction to an incorrect one; a **corrective** transition does the reverse. A **lateral** transition changes one incorrect class to another.

The total parameter count includes 16 fixed rotary-frequency parameters. The checkpoint stores epoch 281 using zero-based indexing, corresponding to reported epoch 282.

## Quick start

Use **Python 3.13** for the analysis environment used by CI. Clone the repository and run the following commands from its root directory.

```bash
git clone https://github.com/DrCanD/spikescr-local-event-geometry.git
cd spikescr-local-event-geometry
```

### Linux and macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python -m pip install --no-build-isolation --no-deps -e .
python -m ssc_geometry verify
python -m ssc_geometry reproduce --out outputs/analysis_01
```

<details>
<summary><strong>Windows PowerShell</strong></summary>

Use the environment's Python executable directly; activation is optional.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-analysis.txt
.\.venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e .
.\.venv\Scripts\python.exe -m ssc_geometry verify
.\.venv\Scripts\python.exe -m ssc_geometry reproduce --out outputs/analysis_01
```

</details>

`verify` checks the release files against the checksum manifest. `reproduce` recalculates benchmark classification metrics, outcome counts, source summaries, bootstrap intervals, internal contrasts, replacement rates and false-discovery-rate adjustments. Recorded summaries are used as comparison targets after calculation.

| Output | Location |
| --- | --- |
| Recomputed statistics | `outputs/analysis_01/statistics/` |
| Full-precision CSV tables | `outputs/analysis_01/tables/` |
| Numerical agreement and validation reports | `outputs/analysis_01/` |

Use a new output directory for each run; existing results are never overwritten. Retain the repository alongside the installed package because the package alone does not include the research data. To run from another location, provide the repository explicitly: `python -m ssc_geometry --root /path/to/repository verify`.

## Model replay

First prepare a separate inference environment using the [reproduction protocol](docs/REPRODUCIBILITY.md#configure-inference-deliberately). The recorded reference uses **PyTorch 2.11.0+cu128 on CUDA**. The protocol also provides a CPU diagnostic setup.

The model source is pinned to upstream commit [`095f418`](https://github.com/JackieWang9811/SpikeSCR/tree/095f418f53b3b24c21caf558225c65ad674d44b1). Eight source files are downloaded and checked against their recorded Git blob hashes before import.

```bash
python scripts/prepare_upstream.py
python -m ssc_geometry checkpoint

# Check four fixed probes, covering all four transition types.
python scripts/run_singleton_audit.py --mode preflight --out outputs/preflight_01

# Evaluate the complete panel and all declared interventions.
python scripts/run_singleton_audit.py --mode full --out outputs/singleton_01
```

A full audit covers all 100 sources, 725,070 candidates and 25,820 sets of seven activation replacements. Newly computed outputs are compared with the released records, and the run stops on a conformance failure.

To resume an interrupted run, repeat the same command with `--resume`. Completed source archives are verified before reuse; an incomplete source is recomputed.

**Batch size is part of the experiment:** the local audit uses one sample per forward pass; the benchmark uses batches of 256. Keep these execution paths separate. See the protocol for [benchmark replay](docs/REPRODUCIBILITY.md#replay-the-benchmark-only-when-intended) and [raw validation-data checks](docs/REPRODUCIBILITY.md#validate-the-raw-validation-input).

## Repository contents

| Path | Contents |
| --- | --- |
| [`src/ssc_geometry/`](src/ssc_geometry/) | Input transforms, neighborhood enumeration, statistics and model-replay helpers |
| [`scripts/`](scripts/) | Upstream-source preparation, singleton audit and benchmark replay |
| [`configs/`](configs/) | Experiment settings, recorded environment and upstream hashes |
| [`data/model/`](data/model/) | Original frozen checkpoint |
| [`data/panel/`](data/panel/) | Transformed validation inputs, source manifest and clean score vectors |
| [`data/neighborhood/`](data/neighborhood/) | Canonical candidate records and source geometry |
| [`data/internal/`](data/internal/) | Activation-change metrics and complete patched score vectors |
| [`data/benchmark/`](data/benchmark/) | Recorded validation and test predictions |
| [`data/reference/`](data/reference/) | Numerical summaries used for comparison |
| [`tests/`](tests/) | Operator, analysis, integrity and model-loading checks |
| [`validation/`](validation/) | Executed checks and their provenance |
| [`checksums/`](checksums/) | File-integrity manifest and SHA-256 listing |

### Artifact coverage

The release includes every candidate's prediction and scalar score descriptors, six activation-change metrics at each boundary, and all **35-class patched score vectors**. It does not store every raw hidden-state tensor or every full unpatched candidate score vector. Candidate replay therefore compares the recorded prediction and score descriptors; clean and patched score vectors are checked in full.

The transformed 100-source validation panel is included. Raw SSC train, validation and test event files are obtained separately when needed. Source hashes and derivation details are documented in the [provenance record](docs/provenance.json).

## Validation

| Check | Recorded status |
| --- | --- |
| File integrity and statistical regeneration | Passed; see the [validation records](validation/README.md). |
| Real-model CPU preflight | Four fixed probes passed, including 64 forward passes and a resume check. |
| Complete model replay in the reference CUDA environment | A full conformance report is not yet included. |

[GitHub Actions](https://github.com/DrCanD/spikescr-local-event-geometry/actions/workflows/validate.yml) runs recorded-result checks and a separate real-model CPU preflight. Its passing status covers those checks, rather than the complete 725,070-candidate replay. Recorded CPU diagnostics include a source-level score discrepancy; their scope is described in the [validation notes](validation/README.md#model-replay). Full CPU and cross-device equivalence have not been established.

The original environment is documented only to the extent recoverable from the experiment records. Use each replay's environment and numerical conformance report to assess its result.

<details>
<summary><strong>Numerical comparison and integrity</strong></summary>

Counts and class outcomes are checked exactly. Statistical comparisons allow `1e-6` for designated float32-derived descriptive means and confidence limits, and `1e-12` for the remaining quantities, including p-values, q-values and recovery rates. Every numerical difference is reported. Model-replay tolerances are specified separately in [`configs/experiment.json`](configs/experiment.json).

Source utterances are the statistical sampling units; candidate moves are nested measurements. A separate implementation recounts the outcome categories, and SciPy independently checks the Spearman coefficient and internal FDR calculations.

SHA-256 checks establish consistency with the included manifest. Identify the repository commit when recording or citing a reproduction run.

To run the test suite in an activated environment:

```bash
python -m unittest discover -s tests -v
```

Tests that require PyTorch or the verified upstream source report a skip when their prerequisites are unavailable. The dedicated real-model CI job requires those prerequisites.

</details>

## Citation

Please cite the associated manuscript and identify the repository commit used in your work:

> İsmail Can Dikmen. *Stable predictions can hide substantial temporal changes in a high-accuracy spiking speech classifier.*

Software citation metadata is available in [`CITATION.cff`](CITATION.cff).

## Attribution and terms

The study uses the [SpikeSCR implementation](https://github.com/JackieWang9811/SpikeSCR) and the Spiking Speech Commands dataset. See [third-party notices](docs/THIRD_PARTY_NOTICES.md) for source references, attribution and dataset terms.

Project-specific code, checkpoint and study results currently have no blanket license assigned. Consult [`LICENSE_NOTICE.md`](LICENSE_NOTICE.md); dataset and upstream-code terms apply separately.
