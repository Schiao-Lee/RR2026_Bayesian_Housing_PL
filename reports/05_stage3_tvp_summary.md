# Stage 3 — Bayesian Hierarchical Time-Varying Parameters (TVP)

**Observations:** 39,112 (stratified 20% subsample by city × snapshot_month, seed=42; 15 cities × 11 months)

**Model:** State-space TVP. Both `α_{c,t}` (city intercept) and `β_{c,t}` (city-specific centreDistance slope) follow non-centered Gaussian random walks across the 11 monthly snapshots. `squareMeters` and `poiCount` slopes are pooled (time- and city-invariant) to keep the parameter count tractable. Gaussian likelihood.

**Sampler:** NUTS via PyMC — 2 chains × 1000 draws (1500 tune), target_accept=0.9, seed=42

## Population / hyperparameter posterior

```
                mean      sd   hdi_3%  hdi_97%  mcse_mean  mcse_sd   ess_bulk   ess_tail   r_hat
grand_alpha  13.1597  0.0788  13.0125  13.3081     0.0042   0.0025   345.1698   629.2042  1.0034
grand_beta   -0.0034  0.0197  -0.0399   0.0331     0.0008   0.0006   567.5931   757.1284  1.0114
gamma_sqm     0.2843  0.0011   0.2822   0.2862     0.0000   0.0000  6602.0600  1609.5876  1.0016
delta_poi     0.0152  0.0014   0.0127   0.0178     0.0000   0.0000  5314.5147  1339.7458  1.0019
sigma         0.2113  0.0008   0.2099   0.2127     0.0000   0.0000  5242.8192  1324.8205  1.0013
sigma_alpha   0.0277  0.0022   0.0234   0.0316     0.0001   0.0000   814.7858  1383.9013  1.0011
sigma_beta    0.0068  0.0015   0.0038   0.0096     0.0000   0.0000  1151.6298  1638.1569  1.0001
tau_alpha     0.3062  0.0663   0.1964   0.4308     0.0026   0.0017   632.1602  1277.3891  1.0015
tau_beta      0.0735  0.0168   0.0466   0.1034     0.0005   0.0006   996.1435  1194.3967  1.0039
```

## City intercept change (α: mean over months [0, 1, 2] vs [8, 9, 10])

| City | α early | α late | Δ log PLN | Δ % (≈) |
|---|---:|---:|---:|---:|
| gdynia | +13.344 | +13.531 | +0.187 | +20.6% |
| krakow | +13.526 | +13.709 | +0.183 | +20.1% |
| czestochowa | +12.704 | +12.863 | +0.159 | +17.2% |
| bialystok | +13.029 | +13.188 | +0.158 | +17.2% |
| katowice | +12.927 | +13.082 | +0.154 | +16.7% |
| warszawa | +13.707 | +13.847 | +0.140 | +15.0% |
| lublin | +13.143 | +13.273 | +0.131 | +14.0% |
| gdansk | +13.498 | +13.622 | +0.125 | +13.3% |
| wroclaw | +13.380 | +13.500 | +0.120 | +12.7% |
| poznan | +13.219 | +13.338 | +0.120 | +12.7% |
| szczecin | +13.044 | +13.153 | +0.109 | +11.6% |
| rzeszow | +13.198 | +13.301 | +0.103 | +10.9% |
| radom | +12.814 | +12.889 | +0.075 | +7.8% |
| lodz | +12.954 | +13.028 | +0.074 | +7.7% |
| bydgoszcz | +12.988 | +13.051 | +0.063 | +6.5% |

## Convergence

- Max R-hat (overall): **1.0123** (target < 1.01)
- Min ESS-bulk (overall): **338**
- Divergences after tuning: **0** / 2000 draws (0.00%)

## Figures

- City intercept trajectories: `notebooks/figures/stage3_city_intercept_trajectories.png`
- City distance-slope trajectories: `notebooks/figures/stage3_city_slope_trajectories.png`

## Comparison anchor

EDA §5 documented a +16% national median price surge between Aug 2023 and Dec 2023, followed by a plateau through Jun 2024. The α_{c,t} trajectory plot should reproduce that as a roughly common drift across cities, with the divergences noted in EDA §5 (Kraków/Gdańsk strong appreciation; Katowice early-2024 decline; Białystok small-N noise). Sign-flip context: the static Stage 2 population centreDistance slope was −0.057; β_{c,t} should fluctuate around that level once city-time variation is allowed.
