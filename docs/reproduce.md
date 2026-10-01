# Reproduction protocol

Three different operations answer three different questions:

1. `python -m ssc_geometry verify` reads every file and checks its hash against `provenance/manifest.json`.
2. `python -m ssc_geometry paper` (and `census`, `search`, `replicas`, `execution`) recomputes the manuscript's
   statistics from the released arrays and compares them with the manuscript values. No model is executed.
3. The programs under `scripts/` execute the neural network and compare fresh outputs with the released arrays.

Passing hashes do not validate a statistic; a regenerated table does not validate a model replay.

## 1. Analysis environment

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements/analysis.txt
python -m pip install --no-build-isolation --no-deps -e .
python -m ssc_geometry verify
python -m ssc_geometry paper --out outputs/paper
python -m unittest discover -s tests
```

Python 3.11–3.13; NumPy 2.3.5 and SciPy 1.17.0 are pinned. The environment in which the reference files were
produced is in `provenance/validation/environment.json`.

### What the commands check

- `census` re-derives every candidate specification, label, horizon and multiplicity from the panel inputs, checks the
  transition partition, the alignment of the internal metrics and replacement scores with the census, the positive
  controls (replacement at stem/block 1/block 2 reproduces the clean score vector), and the 100 source rows of
  `data/neighborhood/sources.csv`. It then recomputes Table 3, the source statistics of §4.2 (margin–adverse
  Spearman ρ with the within-label permutation test, the label-stratified source bootstrap of the adverse rate, the
  cumulative adverse load), the accuracy decomposition of §4.3, Tables 5–8 and the geometry contrasts, and runs an
  independent second route (scalar counting, SciPy Spearman, SciPy BH) over the same quantities.
- `search` recomputes the exact finite-population discovery expectations for every allocation policy and budget,
  rechecks the frozen rankings against the move scores (candidate-index tie break), re-derives the runner-up move
  scores from the input gradients, looks the rankings up in the census for Table 4, computes the post hoc prefix costs,
  and checks the cutoff sweep (`cutoff_sensitivity.csv`), the all-target summary and the witness records.
- `replicas` checks the per-source and paired outcome tables against each other and against the full-validation
  outputs, recomputes Table 9 and the four-source decomposition, and runs the paired within-label source bootstrap
  (20,000 repetitions, seed 20260929: one generator, cohorts in the declared order, labels ascending, one
  `repetitions × quota` index block per label stratum).
- `execution` checks the contracts stored inside the arrays, the alignment of inputs and horizons across conditions,
  the archived benchmark against the fresh batch-256 replay, and recomputes Table 2, the paired contrasts of Table A.1
  (changed labels split into correct→wrong, wrong→correct, wrong→different-wrong), the padding-gap counts of Table A.2,
  the device replay of Table A.3 and the score-difference distributions of Appendix A.3.
- `paper` runs the four and writes `report.json`: for each of the 434 manuscript cells, the regenerated value and
  PASS/FAIL at the printed precision (counts exact; decimals within half a unit of the last printed digit; the
  values of the padding decomposition and pair controls are read from `data/execution/summary.json` and marked
  `kind: summary`).

Statistical units are sources; candidates are nested measurements. Bootstrap and permutation streams keep their
seeds, ordering and repetition counts. Trace means keep the original float32 reduction; `*_numerical_agreement.json`
lists the sub-1e-6 reduction differences between platforms, and no reference value is substituted to force equality.

### Raw validation input

Obtain `ssc_valid.h5.gz` from the [dataset page](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/)
(compressed MD5 `555645b13e15fe270fbfe39cb763f805`) and decompress it; the uncompressed SHA-256 is in
`configs/experiment.json`.

```bash
python -m pip install h5py
python -m ssc_geometry raw-validation --h5 /path/to/ssc_valid.h5
```

The file is hashed before it is opened. Every panel input is then rebuilt with two implementations of the transform
and compared with `data/panel/inputs.npz`. An event exactly on a 5 ms boundary belongs to the last existing bin; do
not replace the transform with a library default.

## 2. Inference environment

The reference runs used `torch 2.11.0+cu128` on CUDA. Install the matching PyTorch build in a separate environment,
then the pinned model dependencies:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0+cu128 torchvision==0.26.0+cu128
python -m pip install -r requirements/inference-cuda.txt
python -m pip install --no-build-isolation --no-deps -e .
python -m pip check
python scripts/prepare_upstream.py        # downloads the 8 pinned SpikeSCR files into .cache/upstream and checks their git blob hashes
python -m ssc_geometry checkpoint         # file and tensor-state hashes, parameter counts; no forward pass
```

`requirements/inference-cuda-pinned.txt` holds the wheel hashes of the three model dependencies for
`pip install --require-hashes`. The full transitive set of the original environment was not saved; record the actual
environment of a replay and judge it by its conformance output, not by version similarity.

The loader (`ssc_geometry.upstream.load_model`) verifies the pinned source, applies the 3-D BatchNorm adapter needed
by the pinned SpikingJelly version, sets the 5 ms input override, loads the checkpoint strictly, checks 3,302,416 total
and 3,302,400 trainable parameters, and resolves the seven boundaries. Weights, neuron equations and the attention
reshape are not changed.

### Canonical audit path

The canonical census uses the unmodified model on CUDA, batch size one, the source's native horizon, a state reset
before every forward, and the public time-summed softmax readout. Batch-256, padded-singleton, CPU-singleton and
q/k-isolated runs are separately identified conditions (Appendix A).

### Full singleton audit

```bash
python scripts/run_audit.py --mode preflight --out outputs/preflight      # 4 fixed probes, one candidate each, all boundaries
python scripts/run_audit.py --mode full --out outputs/audit               # 100 sources, 725,070 candidates, 180,740 replacements
python scripts/run_audit.py --mode full --out outputs/audit --resume      # continue after an interruption
```

For every candidate the prediction, the stored score descriptors, the six metrics at seven boundaries and (for
class-changing candidates) the seven replacement score vectors are compared with the released arrays at the
tolerances in `configs/experiment.json`. A classification disagreement always fails. Completed sources are written
atomically; resume re-validates the stored outputs and the execution digest and never mixes code, configuration or
environments. A report is complete only when it covers all 100 sources, 725,070 candidates and 25,820 replacement
sets.

### Benchmark

```bash
python scripts/run_benchmark.py --split validation --h5 /path/to/ssc_valid.h5 --out outputs/benchmark
```

Batch size 256, ascending source order, zero padding to the group maximum. The official test split needs
`--split test --allow-official-test-replay` and the test file; it is a replication of the single evaluation reported in
Table 2, not a model-selection step.

### Gradient rankings (§4.3)

```bash
python scripts/run_search_gradients.py --step gate runner-up all-target compare --panel --device cuda --out outputs/gradients
```

`gate` reproduces the 100 clean score vectors (tolerance 1e-4, same class) and hashes every LIF spike tensor;
`runner-up` and `all-target` compute one and 34 input gradients per source with the LIF nodes in training mode
(public ATan surrogate) after checking that the surrogate forward is bit-identical to the evaluation forward;
`compare` reports, per source and policy, whether the move scores and rankings equal the retained ones, their
Spearman correlation and top-100 overlap. `--panel` reads the 100 released inputs; `--h5` reads the dataset.
Differences at the 1e-6 level between hardware change the order of near-tied candidates, which `compare` makes visible.

### Execution controls (Appendix A)

```bash
python scripts/run_execution_controls.py --step clean-gate gpu-conditions cpu-singleton isolation padding pairs candidate-replay compare \
    --h5 /path/to/ssc_valid.h5 --gpu cuda --out outputs/execution
```

Each step writes arrays with the same keys as the retained files under `data/execution/`; `compare` reports label
disagreements and maximum score differences between the fresh run and the retained records. See
[execution_controls.md](execution_controls.md).

## 3. CPU runs and the known portability limit

A CPU environment (`requirements/inference-cpu.txt`, torch 2.11.0+cpu) runs the preflight audit and the panel-level
steps of the two scripts:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.11.0+cpu torchvision==0.26.0+cpu
python -m pip install -r requirements/inference-cpu.txt
python scripts/prepare_upstream.py
python scripts/run_audit.py --mode preflight --device cpu --allow-nonreference-environment --out outputs/cpu_preflight
python scripts/run_search_gradients.py --step gate runner-up compare --panel --device cpu --sources 2 --out outputs/cpu_gradients
python scripts/run_execution_controls.py --step clean-gate candidate-replay --panel --gpu cpu --candidate-limit 6 --out outputs/cpu_execution
```

These are smoke tests of the loading, forward, gradient and intervention code, not results. The known limit:
source 4959 keeps its class on the CPU but its clean score vector differs from the reference by 0.3866 (the
tolerance is 1e-5); the record is `provenance/validation/cpu_portability.json`, the cause has not been established,
and no tolerance or reference was changed to accept it. The CPU singleton row of Table 2 comes from a complete CPU run
(`data/execution/validation_cpu_singleton.npz`) and differs from the GPU singleton on 2 of 9,981 labels (Table A.1).

## 4. Scope

The reproduction target is the experiment at the released checkpoint. Training a new model, the replica audit
(whose weights and program are not included) and manuscript figure rendering are outside it. A complete fresh CUDA
conformance report of the full singleton audit is not included; the CPU/GPU replay of the 25,820 class-changing
candidates and the full-validation execution controls are the included model-level evidence.
