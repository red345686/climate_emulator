# Reproducing the scalar-blend paper

This package can run from any directory on a new machine. It does not depend on
an original author account, server path, virtual environment, or the old
`improved_*` folders. Run all commands below from the directory containing this
README and `run_paper.py`.

## 1. Download and arrange the files

**Dataset download link: https://drive.google.com/drive/folders/1LuZIpVrKJYfn-vH2DUhHhdcC9WuWRos3?usp=sharing**

The authors will replace the placeholder above with the shared Drive link.
Download the dataset archive from that link and extract it into `data/` beside
`run_paper.py`. Preserve the internal paths and filenames. The expected layout is:

```text
project/                         # any directory name, on any drive
  README.md
  run_paper.py
  run_branch_study.py
  run.sh
  requirements.txt
  pyproject.toml
  configs/
  src/
  paper_reproduction/
  branch_study/
  tests/
  data/
    raw/                         # inputs_*.nc and outputs_*.nc
    archives/                    # original ClimateBench release archives
    manifests/                   # dataset audit, split and support records
  results/                       # created automatically; may also hold references
    publication-coverage-v2-full-gatefix/   # optional original reference archive
    publication-branch-analysis-v1/        # optional original analysis archive
```

Do not accidentally create `data/data/raw/`. The loader searches for the original
NetCDF filenames recursively under `data/raw/`. The shared data must include
historical, hist-GHG, hist-aer, SSP126, SSP245, SSP370 and SSP585 inputs/targets,
plus the original audit and support records under `data/manifests/`.

The audit may contain original-machine absolute filenames. Preflight now resolves
those filenames under the current raw-data root and checks their original
SHA-256 hashes. The stored audit, data values, split declarations and dataset
identity are not rewritten. Missing, renamed, ambiguous or changed files fail the
checks. No path-editing of JSON records is required.

## 2. Create a fresh environment

Use **Python 3.11**. The primary instructions below use Bash on Linux or WSL2.
For NVIDIA GPU execution, the machine needs an appropriate NVIDIA driver and
TensorFlow-compatible CUDA libraries. See the official TensorFlow installation
instructions: https://www.tensorflow.org/install/pip . TensorFlow GPU support
for this version requires Linux/WSL2 rather than native Windows.

```bash
# Run from the project directory containing this README:
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[neural,test]"
python -m pip check

# Confirm the pinned neural stack and inspect available GPUs:
python -c "import tensorflow as tf, keras; print('TensorFlow:', tf.__version__); print('Keras:', keras.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"

# Save the actual resolved environment for reporting/reproduction:
python -m pip freeze > environment.txt
```

The neural dependencies are TensorFlow 2.15.1 and Keras 2.15.0. Do not upgrade to
Keras 3 or an unpinned current TensorFlow release for this reproduction. If an
NVIDIA GPU was expected but the list is empty, configure that environment before
starting the full run. CPU execution is possible but CNN-LSTM training can be
substantially slower. The classic stack is pinned in `requirements.txt`;
Matplotlib is pinned to 3.9.2 for prebuilt plotting-library availability. It is
not used to fit the climate models or calculate their metrics.

On native Windows, create the environment with `py -3.11 -m venv .venv`, activate
it with `.venv\Scripts\Activate.ps1`, and run the same pip commands. Use
`python run_paper.py` in place of `bash run.sh` in the commands below. Run the
branch analysis with `python run_branch_study.py`. For GPU reproduction, use the
Linux/WSL2 instructions above.

## 3. Run the small tests

```bash
export PYTHONPATH="$PWD/src:$PWD"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

python -m pytest -q paper_reproduction/tests branch_study/tests
bash run.sh --stage plan
```

The tests use synthetic data and do not require the shared dataset or start
full climate-model training. Continue only if they pass.

## 4. Check the downloaded data before training

```bash
bash run.sh --stage preflight \
  --data data/raw \
  --output results/publication-paper-minimal-v1
```

Expected message: `Preflight passed; no fits started`.

All paths are relative to the project directory. If you deliberately store the
data elsewhere, replace `--data data/raw` with its actual path. `archives/` and
`manifests/` must remain siblings of `raw/`. Paths containing spaces must be
quoted. Keep the same paths and configuration throughout one run.

## 5. Reproduce the main study

This trains LPS, ridge, random forest, CNN-LSTM and the scalar blend for both
targets across the four training pools. CNN-LSTM uses the configured seeds and
averages their predictions before scoring. SSP585 is excluded from fitting and
training-side model selection. It is evaluated over 2080–2100.

```bash
bash run.sh --stage all \
  --data data/raw \
  --output results/publication-paper-minimal-v1
```

To run in the background on Linux/WSL2 instead:

```bash
mkdir -p logs
nohup env PYTHONUNBUFFERED=1 STUDY_THREADS=4 \
  bash run.sh --stage all \
  --data data/raw \
  --output results/publication-paper-minimal-v1 \
  > logs/paper_reproduction.log 2>&1 < /dev/null &
echo $! > logs/paper_reproduction.pid
tail -f logs/paper_reproduction.log
```

If necessary, set `CUDA_VISIBLE_DEVICES` to a GPU allocated to you before starting
the run. Ctrl+C in `tail -f` closes the log viewer; it does not stop the background
job. Check completion with:

```bash
python -c "import json; print(json.load(open('results/publication-paper-minimal-v1/status.json')))"
```

For the complete `--stage all` run, both `status: complete` and `stage: all` are
required. Separate `selection`, `final` and `evaluate` stages are available; a
completed individual stage does not mean the complete study has finished.

## 6. Reproduce the three branch analyses

After the main run completes:

```bash
bash branch_study/run.sh \
  --source results/publication-paper-minimal-v1 \
  --data data/raw \
  --output results/publication-branch-analysis-minimal-v1
```

This runs branch-error complementarity, joint BC/SO2-feature controls, and
weight/regularization sensitivity. It is CPU-only and does not retrain CNN-LSTM.
An already downloaded complete original coverage archive can also be used as
`--source results/publication-coverage-v2-full-gatefix` to rerun only the analyses.
Use a fresh output folder for this relocated/refactored code.

## 7. Optional comparison against the original paper results

The original coverage archive is optional for new training. If supplied, extract
the complete folder under `results/` as shown in the directory tree. Then replay
the saved original selections without training:

```bash
bash run.sh --stage audit-reference \
  --reference results/publication-coverage-v2-full-gatefix \
  --output results/publication-paper-minimal-audit
```

Expected: `Original selection replay: 40/40 passed`.

To also compare new metrics with the original metrics, append this option to
**both preflight and the main training command from the start**:

```text
--reference results/publication-coverage-v2-full-gatefix
```

Do not add or remove `--reference` halfway through an existing run; its presence
is part of the run identity. If steps 4–5 were already started without it, use a
fresh output folder such as `results/publication-paper-minimal-comparison-v1`.
With a reference, examine `metrics/original_comparison.csv` and its `matches`
column. Small backend differences can occur; review discrepancies before
replacing any published value. Reference comparison does not load old V4 or
transformer model objects and does not require their code folders.

## 8. Outputs and restart

Keep the complete new folders:

```text
results/publication-paper-minimal-v1/
results/publication-branch-analysis-minimal-v1/
```

They contain score CSVs, selected settings, predictions, fitted models where
applicable, protocol locks, file hashes and status records. Keep
`environment.txt` with the results. The main run produces 40 model/pool/target
prediction cases. Baseline scores retain all four named methods individually.

Repeat the same command to resume with unchanged code, environment, data and
configuration. Completed selections and final cases are hash-verified and reused;
an interrupted case repeats. The main selection checkpoint is written after a
case's folds finish, rather than after each individual fit. If a stopped Linux
job leaves a lock, same-host dead-PID recovery is available with
`--recover-stale-lock`. The recorded PID must no longer exist.

Do not try to resume results from the old code identity, original server location,
or a different runtime. Start a fresh output directory after code/environment
changes or relocation of an already started run. The portable dataset lookup
allows a new run at another location; it does not rewrite old run identities.

## Validation and scope

The original clean package passed eight synthetic tests. The portable-audit
lookup adds a further test. All 40 original selection decisions were replayed
from saved fold statistics, and 120 primary metrics were recomputed from saved
prediction arrays within numerical tolerance (maximum absolute difference about
3.02e-8). These checks are not fresh full-grid training. Fresh full-grid
CNN-LSTM/RF retraining remains to be verified on the new machine.

The package implements the paper's main four-pool scalar-blend/baseline study and
the three branch analyses. It does not regenerate the older V4/transformer
comparators or A/B/C experiments. The study remains exploratory because SSP585
was inspected during development; rerunning on another machine is numerical
reproduction, not an independent climate-scenario test.
