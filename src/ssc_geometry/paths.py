"""Every repository file the package reads, in one place, relative to the repository root.

Each entry names the manuscript element it serves. Paths are joined with
``root / PATH`` at run time, so the package works from any checkout location.
"""
from __future__ import annotations

# configuration -------------------------------------------------------------------------------- §3
EXPERIMENT_CONFIG = 'configs/experiment.json'            # checkpoint, panel and statistical settings
ENVIRONMENT_CONFIG = 'configs/environment.json'          # recorded inference environment
UPSTREAM_CONFIG = 'configs/upstream.json'                # pinned SpikeSCR source files (git blob hashes)

# frozen model --------------------------------------------------------------------------------- §3.2
CHECKPOINT = 'data/model/checkpoint.pt'

# audit panel ---------------------------------------------------------------------------------- §3.3
PANEL_INPUTS = 'data/panel/inputs.npz'                   # 100 transformed sources ([horizon, 140] counts)
PANEL_SOURCES = 'data/panel/sources.csv'                 # source index, label, horizon, hashes
PANEL_CLEAN_SCORES = 'data/panel/clean_scores.npz'       # clean singleton score vectors

# benchmark ------------------------------------------------------------------------------------ Table 2
BENCHMARK = {'validation': 'data/benchmark/validation_batch256.npz',
             'test': 'data/benchmark/test_batch256.npz'}

# complete neighborhood census ----------------------------------------------------------------- §4.2, Tables 3, 7
CANDIDATES = 'data/neighborhood/candidates.npz'          # 725,070 candidate records
SOURCES = 'data/neighborhood/sources.csv'                # 100 source-level summaries

# internal activation metrics and clean-activation replacements -------------------------------- §4.4–4.6, Tables 5, 6, 8
INTERNAL = 'data/internal/activation_metrics_and_replacements.npz'

# search policies ------------------------------------------------------------------------------ §4.3, Table 4, S2
SEARCH_POLICIES = 'data/search/margin_policies.json'
SEARCH_RANKS = 'data/search/candidate_ranks.npz'
SEARCH_SCORES = 'data/search/candidate_scores.npz'
SEARCH_RUNNER_UP_GRADIENTS = 'data/search/runner_up_gradients.npz'
SEARCH_CUTOFF_SENSITIVITY = 'data/search/cutoff_sensitivity.csv'
SEARCH_WITNESSES = 'data/search/witnesses.json'
SEARCH_SUMMARY = 'data/search/summary.json'

# count-readout replicas ----------------------------------------------------------------------- §4.7, Table 9
REPLICA_SOURCE_OUTCOMES = 'data/replicas/source_outcomes.csv'
REPLICA_PAIRED_SOURCES = 'data/replicas/paired_sources.csv'
REPLICA_READOUT_PARTITION = 'data/replicas/readout_partition.csv'
REPLICA_VALIDATION = {'A': 'data/replicas/validation_A.npz', 'B': 'data/replicas/validation_B.npz'}
REPLICA_CONTRACT = 'data/replicas/audit_contract.json'
REPLICA_PROTOCOL = 'data/replicas/protocol.json'
REPLICA_SUMMARY = 'data/replicas/summary.json'
REPLICA_COHORTS = 'data/replicas/cohorts.json'

# execution controls --------------------------------------------------------------------------- §4.8, Appendix A
EXECUTION_GPU_CONDITIONS = 'data/execution/validation_gpu_conditions.npz'
EXECUTION_CPU_SINGLETON = 'data/execution/validation_cpu_singleton.npz'
EXECUTION_QK_ISOLATION = 'data/execution/qk_isolation.npz'
EXECUTION_CANDIDATE_REPLAY = 'data/execution/candidate_replay.npz'
EXECUTION_SUMMARY = 'data/execution/summary.json'

# recorded contracts (kept verbatim; their hash is stored inside the output arrays) ------------
CONTRACT_GPU_CONDITIONS = 'provenance/contracts/execution_gpu_conditions.json'
CONTRACT_ISOLATION_AND_REPLAY = 'provenance/contracts/execution_isolation_and_replay.json'
CONTRACT_GRADIENT_RUNNER_UP = 'provenance/contracts/gradient_runner_up.json'
CONTRACT_GRADIENT_ALL_TARGET = 'provenance/contracts/gradient_all_target.json'

# integrity and provenance --------------------------------------------------------------------- §3.11
MANIFEST = 'provenance/manifest.json'
SHA256SUMS = 'provenance/SHA256SUMS.txt'
RECORDS = 'provenance/records.json'

# reference values: full regenerated statistics and the manuscript cells ----------------------
REFERENCE = {'benchmark': 'data/reference/benchmark.json',
             'neighborhood': 'data/reference/neighborhood.json',
             'geometry': 'data/reference/geometry.json',
             'internal': 'data/reference/internal.json',
             'search': 'data/reference/search.json',
             'replicas': 'data/reference/replicas.json',
             'execution': 'data/reference/execution.json'}
EXPECTED_VALUES = 'data/reference/expected_values.json'

# directories whose contents a command must never write into
PROTECTED_DIRECTORIES = ('data', 'configs', 'src', 'tests', 'provenance', 'docs', 'scripts',
                         'requirements', '.git', '.github')
# directories in which every file must be listed in the integrity manifest
MANIFEST_DIRECTORIES = ('src', 'scripts', 'configs', 'data', 'tests', 'provenance', 'requirements', '.github')
