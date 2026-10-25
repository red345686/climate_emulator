# Minimal reproduction code for the scalar-blend paper

This is a separate paper-only codebase. It contains no improved_* folders,
publication_routes, transformer models, MORB, ClimaX, driver-input study, or
forcing-gate study. The old project is not modified or deleted by this package.

## What is retained

- src/: the reachable shared data, preprocessing, metric, baseline and training
  modules, retaining the original implementations.
- configs/: the original scientific configurations and their includes.
- paper_reproduction/: scalar LPS-ridge blending and the four named baselines:
  LPS, ridge, random forest and CNN-LSTM. No V4 response variants are run.
- branch_study/: complementarity, joint aerosol controls and sensitivity.
- tests/: small shared synthetic fixtures; further tests are within the modules.
- run_paper.py / run.sh and run_branch_study.py: entry points.
- requirements.txt / pyproject.toml: existing dependency specifications.
- VALIDATION.json: local verification results and their limits.

The seven legacy folders can be omitted from this codebase. Their previous
results are not required for a new main run. Previously downloaded original
coverage results can optionally be supplied for comparison and provenance.

## Validation performed

- Eight synthetic tests passed, including an end-to-end classical-model
  selection, fitting, scoring, resume and corruption test, plus branch-study
  integration tests.
- All 40 original baseline/blend selection decisions were replayed from their
  saved fold statistics; the candidate objectives agree.
- All 120 primary metrics were recomputed from the saved paper prediction arrays
  using the retained scorer. Maximum absolute difference: about 3.02e-8, within
  numerical tolerance; these are not new fits.
- No Python imports from improved_* or publication_routes remain.
- Fresh full-grid CNN-LSTM/RF retraining has NOT been validated locally. Use the
  original server Python environment and review original_comparison.csv before
  replacing published numerical results. The original baseline implementations,
  preprocessing, grids, seeds and neural seed-averaging rule are retained.

This implements the main four-pool study and three analyses in the current paper.
It does not regenerate supplementary V4/transformer comparator experiments or
the older A/B/C experiments. Preserve their archived results if publishing those
comparisons; do not claim this package reproduces those additional pipelines.

## Server layout

Extract minimal_paper_reproduction_bundle.zip inside the existing project:

```text
climate_emulator/
  data/                         existing audited data, kept in place
  results/                      existing result archives, kept in place
  minimal_paper_reproduction/    NEW independent code folder
```

Existing data/raw, data/archives and data/manifests are required by the original
integrity preflight. Paths in the dataset audit must still identify those files.
Do not move raw files without rebuilding the corresponding provenance records.
The paper's LaTeX and figures may remain in a separate paper/ directory.

## Use the already activated server venv

No dependency changes are required if using the original successful environment.
requirements.txt lists the classical stack; CNN-LSTM additionally requires the
original TensorFlow/Keras environment (the project specifies 2.15.1/2.15.0).

```bash
cd /DATA/vaneet_2221cs15/climate_emulator
unzip minimal_paper_reproduction_bundle.zip
cd minimal_paper_reproduction

# Lists tasks/settings, with no climate model fitting:
bash run.sh --stage plan

# Small tests, without real data or neural training:
PYTHONPATH="$PWD/src:$PWD" python -m pytest -q paper_reproduction/tests branch_study/tests

# Replays original choices from saved fold statistics; no neural training:
bash run.sh --stage audit-reference \
  --reference ../results/publication-coverage-v2-full-gatefix \
  --output ../results/publication-paper-minimal-audit

# Verifies existing audited data/runtime before training:
bash run.sh --stage preflight \
  --data ../data/raw \
  --output ../results/publication-paper-minimal-v1 \
  --reference ../results/publication-coverage-v2-full-gatefix
```

To rerun model training deliberately:

```bash
mkdir -p logs
nohup env PYTHONUNBUFFERED=1 STUDY_THREADS=4 \
  bash run.sh --stage all \
  --data ../data/raw \
  --output ../results/publication-paper-minimal-v1 \
  --reference ../results/publication-coverage-v2-full-gatefix \
  > logs/paper_reproduction.log 2>&1 < /dev/null &
echo $! > logs/paper_reproduction.pid
```

--reference is optional for new fitting. If supplied, it must remain the same
across preflight, selection, final and evaluation for a given output directory.
The output folder must be fresh and separate from the original results. This
refactor has a new code identity; it cannot resume the old coverage checkpoints.
Within this refactored run, completed selections and final cases are reused.
An interrupted case repeats. Same-host Linux dead-PID lock recovery is available
with --recover-stale-lock; an existing process is never displaced.

The main output is results/publication-paper-minimal-v1. It includes metrics,
selection decisions, all 40 prediction cases, saved models and integrity records.
With --reference, inspect metrics/original_comparison.csv and its matches column.

## Reproduce the three branch analyses

After the new main run has completed:

```bash
bash branch_study/run.sh \
  --source ../results/publication-paper-minimal-v1 \
  --data ../data/raw \
  --output ../results/publication-branch-analysis-minimal-v1
```

Alternatively, --source can point directly to the existing completed
publication-coverage-v2-full-gatefix folder to rerun only the three analyses.
This uses saved original predictions and metadata without importing old model
code. Use a fresh analysis output: the code fingerprint differs from the old run.

## What to keep or omit

For the working paper codebase, keep only this new code directory, the audited
data, paper source/figures, and relevant result archives. Cache directories,
PID files, old experiment ZIPs and unrelated experiment code need not be included
in the distribution.

Keep a separate backup of the original project until server comparison completes.
Do not delete old fitted joblib files or original source code if you still intend
to load or resume those legacy experiments: their serialized class names retain
their old module paths. Existing prediction NetCDF files and score CSVs remain
usable for the paper without loading the old Python models.
