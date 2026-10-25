# Branch analysis: completed outputs

These are post-development diagnostics on reused SSP585 data, not independent confirmation.

- `branch_complementarity.csv`: weighted error cosine, exact blend-error decomposition and diagnostic oracle weight.
- `aerosol_controls.csv`: full frozen blend, GHG-only fixed-settings control, GHG-only training-selected control, and matched-alpha ridge branches.
- `training_sensitivity.csv`: inner-fold objective surfaces; original choices remain fixed.
- `outer_sensitivity.csv`: exploratory outer-score surfaces; do not select a replacement method from them.
- `each_baseline_comparison.csv`: explicit LPS, ridge, RF and CNN–LSTM comparisons for every control and metric.
- `final/*/*/figure_data.npz`: reference/branch/blend fields, map means and annual global series for plotting.

GHG-only controls retain the identical sample keys and masks. They remove BC and SO2 jointly and do not isolate individual aerosol species or establish a causal physical effect.
If the frozen weight is zero, no ridge branch contributes. Its alpha=100 branch diagnostics are explicitly counterfactual.
Complementarity can arise with positively aligned errors; a negative cross term is not required for error reduction.
Compare fixed-settings controls to isolate the input removal at shared alpha/w; compare separately selected controls to compare deployable selection pipelines.

## Frozen-blend reproduction and branch diagnostics

| Pool | Target | Weight | Error cosine | Blend spatial MSE |
|---|---|---:|---:|---:|
| coverage_ssp126_ssp245 | tas | 0.00 | -0.07146183516137307 | 0.26186918 |
| coverage_ssp126_ssp245 | pr | 0.50 | 0.14767907485810403 | 0.071513848 |
| coverage_ssp126_ssp370 | tas | 0.25 | 0.41297031513308935 | 0.16588297 |
| coverage_ssp126_ssp370 | pr | 0.50 | 0.21789954825339794 | 0.055734029 |
| coverage_ssp245_ssp370 | tas | 0.00 | -0.1885828067934072 | 0.12613455 |
| coverage_ssp245_ssp370 | pr | 0.25 | 0.3463810859706792 | 0.060611085 |
| coverage_ssp126_ssp245_ssp370 | tas | 0.00 | -0.10265117247688563 | 0.14069797 |
| coverage_ssp126_ssp245_ssp370 | pr | 0.25 | 0.26485208825318696 | 0.060484171 |
