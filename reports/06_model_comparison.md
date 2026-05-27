# Model Comparison — Stages 1, 2, 3

Apples-to-apples Leave-One-Out cross-validation (LOO-CV, Pareto-smoothed importance sampling) across the three modeling stages. All three models are fit on the **same stratified ~20% subsample** (39,112 rows, seed=42, stratified by city × snapshot_month) so that elpd values are directly comparable.

## LOO-CV ranking

```
                        rank      elpd_loo       p_loo     elpd_diff    weight          se         dse  warning scale
Stage 3 — TVP              0   5232.075757  147.610371      0.000000  0.982515  150.254339    0.000000    False   log
Stage 2 — Hierarchical     1   2929.709188   19.854915   2302.366568  0.015213  150.330966   66.255225    False   log
Stage 1 — Pooled           2 -15838.051893    5.070972  21070.127650  0.002273  131.358826  153.735445    False   log
```

## WAIC cross-check

```
                        rank     elpd_waic      p_waic     elpd_diff    weight          se         dse  warning scale
Stage 3 — TVP              0   5232.437639  147.248488      0.000000  0.982560  150.251598    0.000000    False   log
Stage 2 — Hierarchical     1   2929.769059   19.795045   2302.668581  0.015170  150.330480   66.254516    False   log
Stage 1 — Pooled           2 -15838.039929    5.059008  21070.477569  0.002271  131.358726  153.733847    False   log
```

## Headline

- LOO-CV best model: **Stage 3 — TVP** (stacking weight 98%)
- elpd_loo gain Stage 1 → Stage 2 (add city hierarchy): **+18,767.8** (-15,838 → 2,930)
- elpd_loo gain Stage 2 → Stage 3 (add city × time TVP): **+2,302.4** (2,930 → 5,232)
- Stacking weights — Stage 1: 0.2%, Stage 2: 1.5%, Stage 3: 98.3%. Stage 3 dominates the optimal predictive ensemble.
- Effective parameter counts (`p_loo`) — Stage 1: 5, Stage 2: 20, Stage 3: 148. The TVP's larger p_loo reflects the city × time latent states actively absorbing signal; LOO-CV penalizes overfit, so the elpd gain is net of that complexity cost.
- Comparison figure: `notebooks/figures/model_comparison_loo.png`

The ranking confirms the modeling progression described in README §Modeling Strategy: city-level hierarchy (Stage 2) delivers the largest single jump in predictive density, and adding city × time-varying parameters (Stage 3) provides a further substantive gain that LOO retains after the model complexity penalty. The +13% mean cross-city growth captured by α_{c,t} and the city-specific β_{c,t} heterogeneity (see `reports/05_stage3_tvp_summary.md` and the trajectory figures) carry real out-of-sample weight — not just in-sample fit.
