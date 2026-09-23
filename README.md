# Local event geometry in a frozen spiking speech classifier

Reproduction package for **Stable predictions can hide substantial temporal changes in a high-accuracy spiking speech classifier**, by İsmail Can Dikmen.

This package contains the actual frozen checkpoint, the transformed inputs for the fixed validation panel, the canonical candidate records, and the complete recorded activation metrics and replacement outputs. Its purpose is to let a reader check the reported results without reconstructing the development history.

The replication target is the **technical experiment with the frozen checkpoint**: input transformation, finite-neighborhood enumeration, predictions, internal activation measurements, activation replacement and statistical analysis. Independent retraining and manuscript presentation assets are outside this scope.

File-integrity checks and statistical regeneration have passed. A complete neural-network replay report is not included; the inference commands below perform that comparison. See [the validation records](validation/README.md) for the scope of the checks actually executed.

## Start with the recorded results

Use Python 3.13 for the tested analysis setup. Run these commands from the repository directory.

Windows PowerShell can use the environment directly, without changing its execution policy.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-analysis.txt
.\.venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e .
.\.venv\Scripts\python.exe -m ssc_geometry verify
.\.venv\Scripts\python.exe -m ssc_geometry reproduce --out outputs/analysis_01
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

On Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python -m pip install --no-build-isolation --no-deps -e .
python -m ssc_geometry verify
python -m ssc_geometry reproduce --out outputs/analysis_01
python -m unittest discover -s tests -v
```

An existing output directory is never overwritten. Use a new directory name for a second run. An installed wheel alone does not contain the research data; retain this repository and pass `--root /path/to/repository` when running from a different installation.

The `reproduce` command recalculates benchmark classification metrics, the four outcome categories, source summaries, the margin association, bootstrap intervals, internal contrasts, replacement rates, and false-discovery-rate adjustments. Reference JSON files are used only after calculation, for comparison. They are not copied into the computed outputs.

Computed JSON results are written to `outputs/analysis_01/statistics` and full-precision CSV tables to `outputs/analysis_01/tables`. Numerical comparison details and the validation summary are saved alongside them.

## What is included

| File or directory | Contents |
| --- | --- |
| `data/model/frozen_checkpoint.pt` | Original checkpoint bytes, selected epoch 282. The stored zero-based epoch is 281. |
| `data/panel/inputs.npz` | The 100 transformed validation inputs, stored as uint16 count tensors. |
| `data/panel/manifest.csv` | Source identities, labels, horizons, counts and per-array hashes. |
| `data/panel/clean_scores.npz` | Clean singleton score vectors aligned with the panel. |
| `data/neighborhood/candidate_records.npz` | All 725,070 canonical candidate predictions, move specifications and scalar score descriptors. |
| `data/internal/activation_metrics_and_replacements.npz` | Six metrics at seven boundaries for every candidate, plus all 180,740 recorded patched score vectors. |
| `data/benchmark` | Previously committed validation and test predictions. No raw test events are included. |
| `data/reference` | Recorded summaries used as independent comparison targets. |
| `checksums` | File-integrity manifest and SHA-256 listing. |
| `validation` | Validation results and the provenance of retained preparation checks. |

The internal archive contains **activation-change metrics**, not every raw hidden-state tensor. The candidate archive contains predictions and scalar score descriptors, not every full 35-class unpatched score vector. Full patched score vectors are included. These distinctions matter when deciding what can be checked without running the model again.

## Recorded operating point

| Quantity | Recorded value |
| --- | --- |
| Training seed and selected epoch | 312 and 282 |
| Trainable parameters | 3,302,416 |
| Validation benchmark | 8,617 / 9,981 |
| Test benchmark | 17,247 / 20,382 |
| Unique neighboring inputs | 725,070 |
| Preserved / adverse / corrective / lateral | 699,250 / 5,157 / 9,031 / 11,632 |
| Event-weighted total | 1,659,158 |
| Initially correct sources / adverse-sensitive sources | 84 / 13 |

The benchmark uses batch size 256. The local audit uses batch size one. These execution paths must not be substituted for one another, because the original batched candidate path did not satisfy singleton parity. The reproduced upstream forward pass is kept unchanged, including its tensor reshapes.

## Run new model inference separately

The model is based on [SpikeSCR](https://github.com/JackieWang9811/SpikeSCR), pinned to commit `095f418f53b3b24c21caf558225c65ad674d44b1`. Its source is fetched separately and checked against eight recorded Git blob hashes before import. It is not vendored or relicensed here.

Read [the reproduction protocol](docs/REPRODUCIBILITY.md) before the expensive run. Install the recorded PyTorch CUDA build and the dependencies listed there. Then:

```bash
python scripts/prepare_upstream.py
python -m ssc_geometry checkpoint
python scripts/run_singleton_audit.py --mode preflight --out outputs/preflight_01
python scripts/run_singleton_audit.py --mode full --out outputs/singleton_01
```

The preflight covers four fixed probes, one for each transition type. It is not the full audit. The full command evaluates every declared neighbor and runs all seven replacements for each class-changing candidate. It stops on a conformance failure and leaves the reference files untouched.

For an interrupted full run, use the same arguments and add `--resume`. Completed source archives are verified before reuse; an incomplete source is recomputed. A new environment, execution-code digest or configuration cannot be silently combined with an older run.

A CPU diagnostic must explicitly request `--device cpu --allow-nonreference-environment`. A passing small CPU diagnostic is not full GPU certification. The complete original environment was not recorded beyond the settings retained in `configs/inference_environment.json`; the package does not invent the missing versions.

## Integrity and numerical comparison

File hashes and discrete outcomes are checked exactly. Stored results were reaggregated through the recorded statistical kernels. A separate implementation recounts all transitions; SciPy independently checks the Spearman coefficient and the three internal FDR families.

Some float32-derived means differ at the final floating-point digits on the packaging CPU. The comparison records every difference. A tolerance of 1e-6 applies only to those descriptive means and their confidence limits. Counts and predictions remain exact; p-values, q-values, recovery rates and float64 quantities use a separate 1e-12 comparison. Numerical agreement is not reported as byte equality.

SHA-256 verifies consistency with the supplied manifest. It does not authenticate a maliciously replaced file together with a maliciously replaced manifest. Use the repository commit and an independently archived release digest when the author publishes them.

## Scope and access

The local operator transfers one integrated count unit to the previous or next 5 ms bin within the same transformed feature. It preserves the horizon and total count, permits an occupied destination, and does not preserve the identity of a raw 700-channel event. The audit is complete only within that finite neighborhood and fixed panel.

[Data provenance and third-party terms](docs/THIRD_PARTY_NOTICES.md) identify the public dataset and upstream implementation. The raw SSC train, validation and test event files are not redistributed. The optional raw-validation checker verifies the original HDF5 against the included panel.

The frozen-checkpoint analysis is the replication target. [The recorded training setup](docs/TRAINING.md) explains how that operating point was obtained; repeating training is not a prerequisite for this audit, and independent retraining is not evaluated by this package.
