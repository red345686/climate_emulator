# Branch mechanism and sensitivity study v1

Runs the three requested follow-ups on the completed four-pool coverage experiment, for **both temperature and precipitation**. Uses existing raw data and completed coverage records. CPU only; no neural training, no GPU, no downloads, and no ClimaX dependency.

## Install in a separate directory

Upload `publication_branch_analysis_v1_bundle.zip` to the existing project directory. Your existing virtual environment may stay activated. Run:

```bash
cd /DATA/vaneet_2221cs15/climate_emulator
mkdir publication_branch_analysis_v1
unzip -q publication_branch_analysis_v1_bundle.zip -d publication_branch_analysis_v1
cd publication_branch_analysis_v1
sha256sum -c branch_study/BUNDLE_SHA256SUMS
```

This directory name is new. Do not extract over the completed experiment or the ClimaX directory. No symlinks are required: the commands below use absolute source/data paths.

Check the expected plan without fitting models:

```bash
python run_branch_study.py --stage plan
```

Launch the tests, preflight and all analyses:

```bash
mkdir -p logs
nohup env PYTHONUNBUFFERED=1 \
  STUDY_PYTHON=/DATA/vaneet_2221cs15/climate_emulator/.venv/bin/python \
  STUDY_THREADS=4 \
  bash branch_study/run.sh \
  --source /DATA/vaneet_2221cs15/climate_emulator/results/publication-coverage-v2-full-gatefix \
  --data /DATA/vaneet_2221cs15/climate_emulator/data/raw \
  --output /DATA/vaneet_2221cs15/climate_emulator/results/publication-branch-analysis-v1 \
  > logs/branch_analysis.log 2>&1 < /dev/null &
echo $! > logs/branch_analysis.pid
```

```bash
tail -f logs/branch_analysis.log
```

The output folder is `results/publication-branch-analysis-v1`. Ctrl+C while viewing `tail -f` only stops the log viewer. The background job continues.

## What the analyses do

1. **Branch complementarity.** Refit the actual LPS and selected-alpha ridge branches, reproduce the archived blend predictions and metrics, then compute area-weighted error inner products, error cosine and the exact spatial-error decomposition. Save reference and branch fields for figures. The cosine is not a Pearson correlation. A test-optimal weight is exported only as a diagnostic, never as a new selected model. For pure-LPS cases, the inactive ridge uses alpha=100 as an explicitly counterfactual reference.
2. **Aerosol-feature control.** Compare full inputs to CO2/CH4-only ridge at the same alpha and blend weight. Separately select a GHG-only blend using the original alpha/weight grid and original held-training-scenario folds. This removes BC and SO2 jointly; it does not establish an individual species' effect or physical causality. Training samples, output targets and masks stay matched.
3. **Sensitivity.** Evaluate nine ridge penalties `[.03,.1,.3,1,3,10,30,100,300]` and 13 weights (0 to 1 by .1, plus .25 and .75), for full and GHG-only branches. Export inner objectives separately from exploratory SSP585 scores. Expanded-grid results do not replace the archived main settings or select a new GHG setting.

All eight pool/target decisions are frozen before final outer diagnostics begin. The full pipeline's 21 original candidate objectives are checked against the archive. Named LPS, ridge, random forest and CNN–LSTM comparisons are exported individually for each control and headline metric.

## Compute and reuse

The planned maximum is **468 multi-output ridge fits and 26 LPS fits**: 18 held-training fold jobs plus 8 final jobs. Linear fits are reused across weights, so 13 weights do not multiply the number of fits. The original linear branches are refitted to provide matching-alpha controls and deterministic reproduction; the bundle does not load version-sensitive saved model objects. It reuses existing source predictions/selection records as audit references and reuses its own completed follow-up folds/cases on restart.

Runtime has not been measured on the server. Existing raw data are required, but no new data are needed. Native-grid predictions are held in RAM for one case at a time, so this needs more memory than the tiny synthetic test. Disk space depends on compression of saved 21-year fields. Restricting BLAS threads avoids using all server CPU cores.

For a preflight-only run, use the same arguments with `--stage preflight`. This verifies the source freeze, prediction hashes and raw file fingerprints without fitting regressions. A mismatch stops the run; there is no silent fallback or substitution. Use a fresh output directory if code, packages, inputs or configuration change.

## Restart

Repeat the launch command after an ordinary failure. Verified completed fold/case checkpoints are reused; an interrupted fold is repeated. A forced kill or server restart can leave `.runner.lock`. First confirm no study process is running:

```bash
pgrep -u "$USER" -af '[r]un_branch_study.py'
```

If none is running, add `--recover-stale-lock` to the same launch command. Recovery only removes a lock whose recorded process no longer exists on the same Linux host. It refuses a live process or another host. Do not run two copies against the same output directory.

Status:

```bash
python -c "import json; print(json.load(open('/DATA/vaneet_2221cs15/climate_emulator/results/publication-branch-analysis-v1/status.json')))"
```

## Outputs and plots

- `REPORT.md`: guide and compact branch-diagnostic table.
- `branch_complementarity.csv`: branch alignment and exact error-decomposition quantities.
- `aerosol_controls.csv`: LPS, frozen blend, GHG-only fixed/selected controls and matched-alpha ridge controls.
- `each_baseline_comparison.csv`: individual comparisons against all four original baselines.
- `training_sensitivity.csv`: inner-fold objectives; selection evidence only.
- `outer_sensitivity.csv`: complete outer score surfaces, explicitly exploratory.
- `final/POOL/TARGET/figure_data.npz`: valid-cell fields, mean maps, annual global series, grid and mask.
- `identity.json`, `decisions_freeze.json`, per-fold/case manifests: input/configuration/runtime identity and resume integrity.

After completion, generate PDF figures:

```bash
python -m branch_study.plot /DATA/vaneet_2221cs15/climate_emulator/results/publication-branch-analysis-v1
```

This generates aerosol-control curves, branch-alignment bars and separate inner/outer sensitivity panels for both targets, without refitting. Matplotlib is used only by this optional plotting command. All scoring is complete before plotting. Use the saved fields for geographic maps; reconstruct valid cells using `mask`.

If that optional command reports `No module named matplotlib`, install the project's plotting dependency in the active environment with `python -m pip install matplotlib==3.9.1`, then repeat only the plotting command. The analyses do not need to be rerun. The plotting script could not be exercised locally because that dependency is absent; see `VERIFICATION.md`.

## Interpretation

This is a **post-development diagnostic study**, not independent confirmation. It cannot establish statistical significance, calibrated uncertainty or publication readiness by itself. Do not select a replacement method using outer sensitivity surfaces. Report inactive branches, temperature, all pools, all baseline scores and adverse component tradeoffs. GHG-only ablations can change model capacity as well as input information; interpret them as controlled prediction comparisons.

## Dependencies

Use the existing experiment environment. Required by the runner: NumPy, SciPy, scikit-learn, xarray, a NetCDF backend, PyYAML and pytest for the bundled checks. The original project's pinned requirements are included as a reference; do not reinstall or upgrade them unnecessarily. The runner does not import TensorFlow or PyTorch. Synthetic tests do not access real climate data.
