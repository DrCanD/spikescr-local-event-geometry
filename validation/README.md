# Validation records

These records distinguish numerical analysis checks from execution of the real neural network.

## Current repository update

`maintenance_status.json` records the later repository update, its actual environment, executed test counts and skips. `maintenance_tests.log` contains that run. The added execution-control and extended-analysis reports describe recomputation from retained arrays, without new model inference. The original checks below remain unchanged as historical evidence.

The current CI workflow executes the original statistical regeneration plus `execution-controls` and `extended-analyses`. The real-model job still performs the separate four-probe CPU preflight. See [execution controls](../docs/EXECUTION_CONTROLS.md) for the full-validation and 25,820-candidate recorded inference evidence.

The paired device replay reproduces every original CUDA class-changing label and retains the same 13 sensitive sources on CPU. It does not establish full CPU score equivalence or replay all originally preserved neighbors.

## Original analysis and model-preflight check

- `status.json` identifies the checked code/configuration digest, executed checks and test counts, including skipped tests.
- `analysis_environment.json` records the locally observed Python and package versions.
- `tests_primary.log` contains the original retained unit and integration test output. PyTorch-dependent tests are skipped when PyTorch is unavailable.
- `statistical_regeneration.json` and `numerical_agreement.json` report recomputation from the released arrays and comparison with the recorded numerical references.
- `upstream_source.json` records verification of the eight pinned upstream source files.
- `model_preflight_cpu.json` records the four-probe real-model run, parameter inventory, numerical differences and resume check.
- `cpu_portability_probe.json` records the failing source-4959 CPU diagnostic. The failure remains a portability limit; it is not converted into a passing result.

The integration check covers all 100 sources, 725,070 candidate records, 30,452,940 activation-metric values and 180,740 recorded replacement score vectors. It does not run the real model.

## Retained preparation evidence

`checkpoint_validation.json`, `panel_preparation.json`, `raw_validation_check.json`, `kernel_extraction.json` and `source_audit.json` are historical checks from initial packaging. They are retained because the checkpoint, panel, recorded arrays and extracted numerical/forward kernels are unchanged. They are not claims that the current environment reran the raw-data or tensor-loading checks.

The original packaging snapshot, including superseded logs, is available in Git at commit `9f964bb5cedfbb0e5bda03a2ab15ccf31d2f5221`.

## Model replay

A complete real-model CUDA conformance report is not included. The CPU preflight passed all four fixed probes and 64 forward passes; a broader CPU check encountered a clean-score mismatch at source 4959. The two evidence summaries identify their actual scope and the execution-code digest. Full checkpoint/data hashes are retained independently in the integrity manifest.

`scripts/run_singleton_audit.py --mode full` generates a full report only after comparison of the newly computed predictions, trace metrics and replacement outputs with the released records. The benchmark has a separate replay command. See `docs/REPRODUCIBILITY.md` for both protocols.
