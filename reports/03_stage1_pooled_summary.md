# Stage 1 — Pooled Bayesian Linear Regression

**Observations:** 195,568

**Model:** `log(price) ~ squareMeters_z + centreDistance_z + poiCount_z` (complete pooling, Gaussian likelihood)

**Sampler:** NUTS via bambi/PyMC — 2 chains × 1000 draws (1000 tune), seed=42

## Posterior summary

```
                     mean      sd   hdi_3%  hdi_97%  mcse_mean  mcse_sd   ess_bulk   ess_tail   r_hat
sigma              0.3619  0.0006   0.3608   0.3630        0.0      0.0  2863.1392  1580.9638  1.0009
Intercept         13.4554  0.0008  13.4539  13.4569        0.0      0.0  2173.6728  1481.1485  1.0002
squareMeters_z     0.2825  0.0008   0.2810   0.2840        0.0      0.0  2207.2673  1628.9419  1.0009
centreDistance_z   0.1123  0.0009   0.1105   0.1140        0.0      0.0  2207.4400  1603.4664  1.0027
poiCount_z         0.1177  0.0009   0.1160   0.1195        0.0      0.0  2116.5213  1447.3556  1.0020
```

## Convergence

- Max R-hat: **1.0027** (target < 1.01)
- Min ESS-bulk: **2117**

## Notes

This pooled model serves only as a reference baseline. It cannot capture the city-level heterogeneity (~3× price gap between Warszawa and Częstochowa, see EDA §4) or the temporal dynamics (national +16% surge Aug–Dec 2023, EDA §5) that motivate Stages 2 and 3.
