# Aggregate accuracy conceals concentrated temporal vulnerability in a spiking speech classifier

[![Validation](https://github.com/DrCanD/spikescr-local-event-geometry/actions/workflows/validate.yml/badge.svg?branch=main)](https://github.com/DrCanD/spikescr-local-event-geometry/actions/workflows/validate.yml)
[![DOI](https://zenodo.org/badge/1383291350.svg)](https://doi.org/10.5281/zenodo.22936264)

Data, code and complete outcome maps behind the manuscript of the same title by **İsmail Can Dikmen**.
Every table and figure of the manuscript is regenerated from the files in this repository by one command,
without a GPU and without executing the neural network.

[What was done](#what-was-done) · [Regenerate the tables](#regenerate-the-tables) · [What is where](#what-is-where) ·
[Glossary](#glossary) · [Run the model again](#run-the-model-again) · [What is not included](#what-is-not-included) · [Citation](#citation)

## What was done

A frozen [SpikeSCR](https://github.com/JackieWang9811/SpikeSCR) checkpoint for Spiking Speech Commands (SSC) was
audited on a fixed panel of 100 validation utterances. For each utterance, **every** distinct move of one spike count
to an adjacent 5 ms bin in the same feature was evaluated: **725,070 neighbors**. Each neighbor's outcome was classified
as class-preserved, adverse, corrective or lateral; its internal activation change was measured at seven boundaries;
and for every class-changing neighbor the clean activation was substituted back at each boundary (**180,740
replacement score vectors**). Search policies, two independently trained count-readout replicas and a set of execution
controls complete the study.

### Main findings

- **Vulnerability is concentrated.** 13 of the 84 initially correct sources admit adverse neighbors; five sources carry
  **93.21%** of all adverse candidates, while the equal-source expected accuracy under a uniform neighbor draw rises
  from 84.00% to 84.54% (Table 3, §4.2, Fig. 3).
- **Informed search does not remove the cost of enumeration.** At 8,400 candidate queries, uniform sampling finds an
  expected 7.61 of the 13 vulnerable sources, the best margin-focused allocation 9.42, and the gradient rankings 7–8.
  Finding all 13 with the runner-up gradient ranking needs a common prefix of 5,612 candidates per source:
  **461,380 queries, 73.45%** of the census of initially correct sources (Table 4).
- **Decision stability and representation stability differ.** 617,941 of the 699,250 class-preserved neighbors change
  the final-block activation (median relative L2 change 0.6464). Among nonzero responses, the activation norm only
  weakly ranks margin displacement (Tables 5 and 7). Replacing a candidate's activation by the clean activation at a
  single attention or local boundary restores the clean, correct decision for 78–93% of adverse candidates in the
  equal-source sense (Table 8).
- **Near-equal replica accuracies hide a 5.21-fold difference** in class-changing neighbors between two count-readout
  replicas (72.28% vs 72.53% validation accuracy). Four sources drive it; on the 67 sources both replicas classify
  correctly the ordering of adverse rates reverses (Table 9).
- **Execution settings define the evaluated function.** Native singleton and batch-256 inference of the same
  checkpoint differ on **619 of 9,981** validation labels; reversing the batch order changes 531. The cause is a
  reshape in the public attention code that mixes sources across the LIF membrane lanes; isolating sources at those
  two boundaries makes all 9,981 batched score vectors bitwise order-invariant (Table 2, Appendix A).

### At a glance

| Item | Value |
| --- | --- |
| Input | SSC, 35 classes; 700 channels integrated into 140 features; 5 ms bins; source-specific horizon |
| Frozen checkpoint | seed 312, epoch 282; 3,302,416 parameters |
| Validation accuracy | 86.3340% (batch 256), **86.0836%** (native GPU singleton, the audit path), 86.0735% (CPU singleton) |
| Official test accuracy | 84.6188% (batch 256, evaluated once) |
| Panel | 100 sources, 84 initially correct, 13 adverse-sensitive, 25 with any class change |
| Neighbors | 699,250 preserved / 5,157 adverse / 9,031 corrective / 11,632 lateral |
| Replicas | 1,297,705 neighbors each; 3,564 vs 18,566 class changes |
| CPU/GPU replay | 25,794 of 25,820 class-changing candidates keep their label; same 13 sources on both devices |

## Regenerate the tables

Python 3.11–3.13, NumPy and SciPy; no GPU, no dataset download, about 20 seconds.

```bash
git clone https://github.com/DrCanD/spikescr-local-event-geometry.git
cd spikescr-local-event-geometry
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install -r requirements/analysis.txt
python -m pip install --no-build-isolation --no-deps -e .
python -m ssc_geometry verify                           # every file against provenance/manifest.json
python -m ssc_geometry paper --out outputs/paper        # every table, figure data and report.json
```

`outputs/paper/report.json` lists **434 numbers printed in the manuscript** (Tables 2–9, A.1–A.3 and the in-text
values), each with the regenerated value and PASS/FAIL at the manuscript's printed precision. The same directory holds
`table_02.csv` … `table_09.csv`, `table_A1.csv` … `table_A3.csv` and `figure_02.csv` … `figure_04_b.csv`.

| Command | Manuscript | Output files |
| --- | --- | --- |
| `python -m ssc_geometry census --out DIR` | §4.1–4.2, 4.4–4.6; Tables 3, 5, 6, 7, 8; Figs. 2–4 | `table_03.csv`, `table_05.csv` … `table_08.csv`, `figure_*.csv`, `sources.csv`, `census.json` |
| `python -m ssc_geometry search --out DIR` | §4.3; Table 4; Supplementary S2 | `table_04.csv`, `search_prefix_costs.csv`, `search_first_adverse_ranks.csv`, `search_cutoff_sensitivity.csv`, `search.json` |
| `python -m ssc_geometry replicas --out DIR` | §4.7; Table 9 | `table_09.csv`, `replicas_paired_bootstrap.csv`, `replicas_four_source_decomposition.csv`, `replicas.json` |
| `python -m ssc_geometry execution --out DIR` | §4.1, 4.8; Tables 2, A.1, A.2, A.3 | `table_02.csv`, `table_A1.csv` … `table_A3.csv`, `execution_score_distributions.csv`, `execution.json` |
| `python -m ssc_geometry paper --out DIR` | all of the above | all of the above plus `report.json` and `report.csv` |

Every command first checks all file hashes, recomputes its statistics from the arrays (never from the reference
files), compares them with `data/reference/`, and writes full-precision CSV tables. Output directories are created
fresh; an existing directory is never overwritten. From another directory use `python -m ssc_geometry --root
/path/to/repository …`. The tests run with `python -m unittest discover -s tests`.

## What is where

| Directory | Contents | Manuscript |
| --- | --- | --- |
| `data/model/` | the frozen checkpoint (`checkpoint.pt`) | §3.2 |
| `data/panel/` | the 100 transformed panel inputs, their labels and hashes, the clean singleton scores | §3.3 |
| `data/benchmark/` | batch-256 validation and test predictions | Table 2 |
| `data/neighborhood/` | all 725,070 candidate records (`candidates.npz`) and the 100 source summaries (`sources.csv`) | §4.2, Tables 3, 7, Fig. 3 |
| `data/internal/` | six activation-change metrics at seven boundaries for every candidate; 180,740 replacement score vectors | §4.4–4.6, Tables 5, 6, 8, Fig. 4 |
| `data/search/` | frozen gradient rankings and move scores, runner-up input gradients, margin-policy specification, cutoff sweep, witness checks | §4.3, Table 4, S2 |
| `data/replicas/` | per-source outcomes of replicas A and B, paired cohorts, readout partition, full-validation scores, audit contract/protocol/summary | §4.7, Table 9 |
| `data/execution/` | full-validation scores under eight execution conditions, q/k-isolation scores, 25,820 CPU/GPU candidate replays | §4.1, 4.8, Appendix A |
| `data/reference/` | the regenerated statistics the commands are compared with, and `expected_values.json` (the manuscript cells) | — |
| `configs/` | experiment settings, pinned upstream source hashes, reference environment | §3 |
| `provenance/` | `manifest.json` / `SHA256SUMS.txt` (every file), the four contracts of the model runs (kept verbatim), `records.json` (file history), validation records | §3.11 |
| `scripts/` | model-executing programs: upstream download, benchmark, full singleton audit, gradient rankings, execution controls | §3 |
| `src/ssc_geometry/` | the package: `census`, `search`, `replicas`, `execution`, `tables` (manuscript cross-check), `statistics`, `core` (transform and move operator), `forward`, `inference`, `upstream` | — |
| `docs/` | [data](docs/data.md) (every file and array key), [reproduce](docs/reproduce.md), [execution controls](docs/execution_controls.md), [checkpoint](docs/checkpoint.md), [release](docs/release.md), [third party](docs/third_party.md) | — |

## Glossary

- **Source** — one validation utterance, identified by its row index in the official SSC validation split.
- **Panel** — the 100 sources audited (label-stratified, chosen before any outcome was known); 84 are *initially
  correct* (clean prediction equals the label).
- **Neighbor / candidate** — the input obtained by moving one spike count from a bin to the adjacent bin (earlier or
  later) in the same feature; `from_bin`, `to_bin`, `feature`. *Multiplicity* is the count at the origin cell.
- **Transition** — `transition_code` 0 class-preserved, 1 adverse (correct → wrong), 2 corrective (wrong → correct),
  3 lateral (wrong → different wrong). Corrective and lateral changes are not errors introduced by the move.
- **Clean score / margin** — scores are the time-summed softmax probabilities (the public readout); the margin is the
  gap between the two largest scores.
- **Boundary** — stem, attention 1, local 1, block 1, attention 2, local 2, block 2 (the module outputs at which
  activations are measured and replaced).
- **Replacement** — a forward pass in which the candidate's activation at one boundary is replaced by the clean
  activation; "restored" means the output returns to the clean prediction.
- **Execution condition** — native singleton (the audit path), padding-matched singleton, batch 256, reversed batch 256,
  q/k-isolated batch 256, and the CPU singleton.
- **Replica A / B** — two count-readout spiking networks trained with different seeds (341969035 / 554720563),
  audited with their native 10 ms neighborhoods; never pooled with the SpikeSCR census.

## Run the model again

The model-executing scripts need a separate environment with PyTorch (the reference runs used
`torch 2.11.0+cu128`), the pinned SpikeSCR source (downloaded and hash-checked by `scripts/prepare_upstream.py`) and,
for the full validation set, the official `ssc_valid.h5`. See [docs/reproduce.md](docs/reproduce.md).

```bash
python scripts/prepare_upstream.py                                      # 8 pinned source files, hash-checked
python -m ssc_geometry checkpoint                                       # checkpoint hashes and parameter counts
python scripts/run_audit.py --mode preflight --out outputs/preflight    # 4 fixed probes
python scripts/run_audit.py --mode full --out outputs/audit             # all 725,070 candidates + 180,740 replacements
python scripts/run_benchmark.py --split validation --h5 ssc_valid.h5 --out outputs/benchmark
python scripts/run_search_gradients.py --step gate runner-up all-target compare --panel --out outputs/gradients
python scripts/run_execution_controls.py --step clean-gate gpu-conditions cpu-singleton isolation padding pairs candidate-replay compare --h5 ssc_valid.h5 --out outputs/execution
```

Each script writes a contract (checkpoint and source hashes, environment) into its output directory, resumes after an
interruption, and compares its outputs with the retained records. The CPU preflight of the audit and the panel-level
steps of the two later scripts also run on a CPU; this is a smoke test, not a result. One CPU portability limit is
known: source 4959 keeps its class but its score vector differs from the reference by 0.3866, so CPU runs are
diagnostics, not the reference.

## What is not included

- The raw SSC event files (obtain `ssc_valid.h5` from the [dataset page](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/);
  the official test set was opened once for Table 2 and is never read by any command here).
- The SpikeSCR source (downloaded from the pinned commit by `scripts/prepare_upstream.py`).
- The raw hidden tensors of the census and the full 35-class score vectors of unpatched candidates (their descriptors are in
  `candidates.npz`); the full all-target gradient tensors (their move scores and rankings are in `data/search/`).
- The weights and raw candidate maps of replicas A and B, and the program that ran their audit (its hashes are in
  `data/replicas/audit_contract.json`); their per-source outcomes are complete.
- The per-source arrays of the padding decomposition and the three pair controls of Appendix A (their summaries are in
  `data/execution/summary.json`; `scripts/run_execution_controls.py` regenerates the arrays).

## Citation

> İsmail Can Dikmen. *Aggregate accuracy conceals concentrated temporal vulnerability in a spiking speech classifier.* Manuscript, 2026.

Software metadata is in [`CITATION.cff`](CITATION.cff). Version 1.1.0 (this package, commit `ea15b24`) is archived as
[10.5281/zenodo.23089436](https://doi.org/10.5281/zenodo.23089436); the concept DOI
[10.5281/zenodo.22936264](https://doi.org/10.5281/zenodo.22936264) always resolves to the latest version. See
[docs/release.md](docs/release.md).

## Attribution and terms

The study uses the [SpikeSCR implementation](https://github.com/JackieWang9811/SpikeSCR) (Wang et al., Neural Networks 195,
108253, 2026) and the SSC dataset (Cramer et al., IEEE TNNLS 2022); see [docs/third_party.md](docs/third_party.md).
Project software is licensed under [MIT](LICENSE); project-authored research outputs, documentation and the trained
checkpoint under [CC BY 4.0](LICENSES/CC-BY-4.0.txt); see [LICENSE_NOTICE.md](LICENSE_NOTICE.md) for the file-level scope.
