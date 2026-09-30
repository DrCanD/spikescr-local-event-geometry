# Recorded experiment sources

These files preserve the sources used for the retained experiment records.
They are provenance evidence, with byte hashes in the run contracts and the
repository manifest. They are not relocated, drop-in versions of the current
repository's inference commands.

| File | Original execution context |
| --- | --- |
| `gpu_execution_controls_20260930.py` | Expanded experiment directory with frozen assets and the full validation cache |
| `mechanism_device_controls_20260930.py` | Colab project on Drive, previous GPU-control run and frozen audit; includes visible progress and resumable controls |
| `run_singleton_gradient_v2.py` | Frozen experiment directory with pinned model source, panel, checkpoint and validation inputs |
| `run_multitarget_20260930.py` | Expanded closure directory and the frozen single-target runner; one source per forward, 34 target backwards per source |

Use `python -m ssc_geometry execution-controls` and `extended-analyses` to
recompute the included records without those historical directory layouts.
Use the top-level `scripts/run_singleton_audit.py` for a fresh canonical model
audit. Raw SSC validation events must be supplied separately for full-validation
inference. No script runs when this directory is imported or listed.
