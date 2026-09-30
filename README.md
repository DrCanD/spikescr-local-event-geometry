# SpikeSCR local event geometry and execution controls

[![Validation](https://github.com/DrCanD/spikescr-local-event-geometry/actions/workflows/validate.yml/badge.svg?branch=main)](https://github.com/DrCanD/spikescr-local-event-geometry/actions/workflows/validate.yml)
[![DOI](https://zenodo.org/badge/1383291350.svg)](https://doi.org/10.5281/zenodo.22936264)

Research code and retained results for **Aggregate accuracy conceals concentrated temporal vulnerability in a spiking speech classifier** by **İsmail Can Dikmen**.

[Findings](#main-findings) · [Quick start](#quick-start) · [Execution contract](#execution-contract) · [Model replay](#model-replay) · [Data coverage](#data-coverage) · [Citation](#citation)

## Overview

Aggregate accuracy does not describe where temporal vulnerability lies, how much internal activity changes, or which function a batched implementation actually evaluates.

This study audits a frozen [SpikeSCR](https://github.com/JackieWang9811/SpikeSCR) checkpoint on Spiking Speech Commands. It evaluates every distinct one-count transfer to an adjacent **5 ms bin**, within the same feature and input horizon, for a fixed panel of 100 validation utterances. The retained outcome map supports exact finite-population search comparisons and source-level statistics. Internal measurements and activation replacement distinguish stable decisions from stable representations. Execution controls separate padding, batch order and device effects.

The census contains **725,070 unique neighbors**, six activation-change metrics at seven boundaries, and **180,740 complete 35-class activation-replacement score vectors**. Completeness refers to the declared neighborhood and panel.

## Main findings

- **Vulnerability is concentrated.** Thirteen of 84 initially correct sources admit adverse transitions. Five sources carry **93.21%** of the adverse candidates, even while source-equal expected accuracy rises from 84.00% to 84.54% under uniform neighbor selection.
- **Informed search still incurs substantial cost.** The retained runner-up gradient ranking needs a common prefix of 5,612 candidates to find all 13 sensitive sources: **461,380 queries, or 73.45%** of the initially correct-source census. This prefix is a retrospective cost calculation; it does not provide the outcome densities obtained by enumeration.
- **Decision and representation stability differ.** Large internal changes also occur when the predicted class is preserved. Branch replacement rates depend on whether sources or candidates receive equal weight; positive full-output replacements are pipeline controls.
- **Nearly equal replica accuracies can mask a concentrated gap.** Secondary count-readout replicas achieve 72.28% and 72.53% validation accuracy but differ **5.21-fold** in class-changing candidates. Four sources drive the contrast; the adverse-rate ordering reverses within their common clean-correct cohort.
- **Execution settings define the evaluated function.** Native CUDA singleton and batch-256 inference differ on **619 of 9,981 labels**. Reversing batch order changes 531 labels. Isolating logical sources at the two q/k LIF boundaries in each attention block removes every order-dependent score difference. Against padding-matched singleton inference, one label difference remains; score equivalence is not established.

## Experiment at a glance

| Item | Recorded result |
| --- | --- |
| Input | SSC, 35 classes; 700 channels integrated into 140 features; 5 ms bins |
| Frozen checkpoint | Seed 312; selected epoch 282, stored zero-based epoch 281 |
| Parameters | 3,302,416 total; 3,302,400 trainable; 16 fixed |
| Validation benchmark, CUDA batch 256 | 8,617 / 9,981 correct, 86.3340% |
| Validation audit path, native CUDA singleton | 8,592 / 9,981 correct, **86.0836%** |
| Validation CPU singleton control | 8,591 / 9,981 correct, 86.0735% |
| Original test benchmark, CUDA batch 256 | 17,247 / 20,382 correct, 84.6188% |
| Local panel | 100 sources, 84 initially correct, 13 adverse-sensitive |
| Preserved / adverse / corrective / lateral | 699,250 / 5,157 / 9,031 / 11,632 |
| Event-weighted denominator | 1,659,158 |
| Paired CPU/GPU replay | 25,794 / 25,820 labels agree, **99.8993%**; same 13 sensitive sources |

**Preserved** means the predicted class stays the same. **Adverse** changes a correct prediction to a wrong one; **corrective** changes a wrong prediction to the true class; **lateral** changes one wrong class to another. A source is one validation utterance; a candidate is one distinct neighboring input. Candidate moves are nested measurements, not independent statistical samples.

## Quick start

Use **Python 3.13** for the analysis environment used by CI. These commands recompute results from included records with visible phase updates. They require no GPU or raw SSC download.

```bash
git clone https://github.com/DrCanD/spikescr-local-event-geometry.git
cd spikescr-local-event-geometry
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python -m pip install --no-build-isolation --no-deps -e .
python -m ssc_geometry verify
python -m ssc_geometry reproduce --out outputs/analysis_01
python -m ssc_geometry extended-analyses --out outputs/extended_01
python -m ssc_geometry execution-controls --out outputs/execution_01
```

On Windows PowerShell, create the environment with `python -m venv .venv` and substitute `.\.venv\Scripts\python.exe` for `python`; activation is optional.

| Command | Recomputed evidence |
| --- | --- |
| `verify` | Every manifest hash and protected repository path |
| `reproduce` | Original benchmark metrics, census, source statistics, internal contrasts, bootstrap/FDR and activation-replacement rates |
| `extended-analyses` | Margin cutoff sensitivity, frozen single/multi-target gradient discovery, capped search costs and replica cohort decomposition |
| `execution-controls` | Full-validation padding/order/device contrasts, q/k isolation and all original class-changing candidate replays |

Each analysis writes JSON reports and full-precision CSV tables to its specified directory. Existing result directories are never overwritten. Retain the repository alongside the installed package: the Python package alone does not contain the research data. From another directory, use `python -m ssc_geometry --root /path/to/repository verify`.

## Execution contract

The **canonical decision census uses the unmodified public model, CUDA, batch size one, native input horizon, a reset before each forward, and the public time-summed softmax readout**. The batch-256 benchmark, CPU singleton replay and source-isolated diagnostic are separately identified conditions.

Padding changes both the readout support and valid-prefix processing. On the same GPU, padding alone changes 344 validation labels. Q/k source isolation removes the 531 original batch-order changes and reproduces padding-matched singleton labels on **9,980 of 9,981 inputs**. The residual score displacement has median 1.91e-6 but maximum **3.7136**; the maximum is not a machine-precision discrepancy.

CPU/GPU candidate replay covers all **25,820 originally class-changing candidates**. All 13 sensitive sources and their first canonical adverse witnesses persist on both devices. The 699,250 originally class-preserved candidates were not replayed in this device control, so additional adverse incidence outside that cohort is not estimated.

The checkpoint was trained using the public coupled batch path. Source isolation is a diagnostic intervention that preserves the public singleton convention; a conventional head/time permutation would define a third function for which these weights were not optimized and which was not evaluated. See [execution controls](docs/EXECUTION_CONTROLS.md) for mechanisms, paired counts, score distributions and control limits.

## Model replay

Create a separate inference environment following the [reproduction protocol](docs/REPRODUCIBILITY.md#configure-inference-deliberately). The recorded canonical environment uses **PyTorch 2.11.0+cu128 on CUDA**. Eight source files are pinned to upstream commit [`095f418`](https://github.com/JackieWang9811/SpikeSCR/tree/095f418f53b3b24c21caf558225c65ad674d44b1) and hash-checked before import.

```bash
python scripts/prepare_upstream.py
python -m ssc_geometry checkpoint
python scripts/run_singleton_audit.py --mode preflight --out outputs/preflight_01
python scripts/run_singleton_audit.py --mode full --out outputs/singleton_01
```

The full command compares new model outputs with all 100 sources, 725,070 candidates and 25,820 seven-boundary replacement sets. Repeat the same command with `--resume` after interruption; verified completed source archives are reused. A complete fresh CUDA conformance report for this entire workload is **not included**. The later class-changing device replay is narrower evidence.

Benchmark replay retains its batch size, order and padding; it does not substitute for the canonical audit. The [training record](docs/TRAINING.md) documents checkpoint provenance. Independently retraining the models and preparing manuscript presentation assets remain separate workflows.

## Data coverage

| Path | Included material |
| --- | --- |
| [`data/model/`](data/model/), [`data/panel/`](data/panel/) | Frozen checkpoint, transformed 100-source panel and full clean scores |
| [`data/neighborhood/`](data/neighborhood/) | Every canonical candidate record and source summary |
| [`data/internal/`](data/internal/) | Six metrics at seven boundaries and full patched score vectors |
| [`data/benchmark/`](data/benchmark/) | Recorded validation and test predictions |
| [`data/search_policies/`](data/search_policies/) | All frozen policy scores/ranks, runner-up input gradients, contracts and witness verification |
| [`data/replicas/`](data/replicas/) | Source-level count-readout records, paired cohorts and full-validation scores |
| [`data/execution_controls/`](data/execution_controls/) | CPU/GPU full-validation scores, q/k isolation outputs, paired candidate scores and original run contracts/summaries |
| [`data/reference/`](data/reference/) | Numerical comparison targets |
| [`scripts/recorded/`](scripts/recorded/) | Byte-preserved experiment sources and their original execution context |
| [`validation/`](validation/), [`checksums/`](checksums/) | Executed checks, provenance and integrity manifests |

The compact search archive preserves policy scores and ranks exactly. All-target gradients and directional gains were independently checked when extracting those ranks; their full tensors are not duplicated here. The original raw hidden tensors and all unpatched census score vectors are also outside the original package. Padding-prefix ablations and three pair controls currently have recorded summaries; their full per-source diagnostic arrays are not bundled. These boundaries are detailed in the [data schema](docs/DATA_SCHEMA.md) and [provenance](docs/provenance.json).

Raw SSC event files are obtained separately. The included full-validation input hashes and horizons establish the pairing of the recorded execution conditions; they are not a substitute for raw inputs when running fresh inference.

## Validation

[GitHub Actions](https://github.com/DrCanD/spikescr-local-event-geometry/actions/workflows/validate.yml) runs all retained-result checks, both added analysis commands and a separate four-probe real-model CPU preflight. Its passing status covers that workload. Historical checks and the source-4959 CPU score discrepancy are retained in the [validation records](validation/README.md).

```bash
python -m unittest discover -s tests -v
```

PyTorch-dependent tests report a skip when their prerequisites are absent. The dedicated real-model CI job installs those prerequisites. Predictions and counts are checked exactly; float32 descriptive statistics and model replay have their separately documented tolerances. Hash checks establish consistency with the supplied manifest, so retain the commit identifier with any reproduction.

## Related work

Yan, Zhu, Tang, Cai, Liu and Wong, [*Latency and accuracy tradeoffs in Spiking Neural Networks*](https://doi.org/10.48550/arXiv.2609.35260), arXiv preprint, 2026. Their construction shows that varying a layer's delay can preserve its spike counts while changing downstream firing and producing non-monotone accuracy. The present study measures a distinct, finite input-retiming neighborhood in a trained classifier. Their hardware reconstruction also motivates making the implemented time-axis convention explicit; it does not validate the batch layout examined here.

## Citation

> İsmail Can Dikmen. *Aggregate accuracy conceals concentrated temporal vulnerability in a spiking speech classifier.* Manuscript, 2026.

Software metadata is in [`CITATION.cff`](CITATION.cff). Identify the repository commit as well as the archive version you use.

The [concept DOI](https://doi.org/10.5281/zenodo.22936264) groups published versions. [Original archive DOI 10.5281/zenodo.22936265](https://doi.org/10.5281/zenodo.22936265) covers the original technical package with software version `0.1.0rc1`; Git tags `v0.1.0rc1` and `v1.0` identify the same original commit. **The later execution controls, search extensions and replica analysis on current `main` are not included in that original DOI version.** Current development metadata uses `0.2.0.dev0`; no new release or archival DOI is asserted. See [release status](docs/RELEASING.md).

## Attribution and terms

The study uses the [SpikeSCR implementation](https://github.com/JackieWang9811/SpikeSCR) and SSC dataset. Retain the [third-party attributions](docs/THIRD_PARTY_NOTICES.md).

Original project software is licensed under [MIT](LICENSE). Project-authored research outputs, documentation and frozen model weights use [CC BY 4.0](LICENSES/CC-BY-4.0.txt), within the author's rights; transformed SSC material retains its dataset attribution. See [LICENSE_NOTICE.md](LICENSE_NOTICE.md) for file-level scope. Third-party software retains its own terms.
