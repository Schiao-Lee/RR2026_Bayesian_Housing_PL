"""
Script: 01_pooled_baseline.py
Stage 1 — Pooled Bayesian Linear Regression baseline.

Ignores city and time structure (complete pooling). Fits:
    log(price) ~ squareMeters_z + centreDistance_z + poiCount_z

This is the simplest Bayesian baseline for the modeling progression described in
README.md. Subsequent stages add city-level hierarchy (Stage 2) and time-varying
coefficients (Stage 3); model comparison via WAIC/LOO will quantify the gain.

Feature selection follows the EDA findings in reports/02_eda_findings.md:
- log(price) as response (raw price is right-skewed; log ≈ Normal)
- squareMeters: strongest predictor (Pearson r = 0.64)
- centreDistance: theoretically core (will become TVP slope in Stage 3)
- poiCount: complementary spatial signal (r = 0.18, low collinearity with the above)
- rooms dropped: multicollinearity with squareMeters (r = 0.83)
"""

from pathlib import Path

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "master_sales_dataset.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "03_stage1_pooled_summary.md"
TRACE_PATH = PROJECT_ROOT / "data" / "traces" / "stage1_pooled.nc"

FEATURES = ["squareMeters", "centreDistance", "poiCount"]


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["price", *FEATURES]).copy()
    df["log_price"] = np.log(df["price"])
    for col in FEATURES:
        df[f"{col}_z"] = (df[col] - df[col].mean()) / df[col].std()
    return df


def fit_model(df: pd.DataFrame) -> az.InferenceData:
    formula = "log_price ~ squareMeters_z + centreDistance_z + poiCount_z"
    model = bmb.Model(formula, df, family="gaussian")
    idata = model.fit(
        draws=1000,
        tune=1000,
        chains=2,
        target_accept=0.9,
        random_seed=42,
        progressbar=False,
        idata_kwargs={"log_likelihood": True},
    )
    return idata


def write_report(idata: az.InferenceData, n_obs: int) -> None:
    summary = az.summary(idata, kind="all", round_to=4)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("# Stage 1 — Pooled Bayesian Linear Regression\n\n")
        f.write(f"**Observations:** {n_obs:,}\n\n")
        f.write(
            "**Model:** `log(price) ~ squareMeters_z + centreDistance_z + poiCount_z` "
            "(complete pooling, Gaussian likelihood)\n\n"
        )
        f.write("**Sampler:** NUTS via bambi/PyMC — 2 chains × 1000 draws (1000 tune), seed=42\n\n")
        f.write("## Posterior summary\n\n```\n")
        f.write(summary.to_string())
        f.write("\n```\n\n")
        f.write("## Convergence\n\n")
        max_rhat = float(summary["r_hat"].max())
        min_ess = float(summary["ess_bulk"].min())
        f.write(f"- Max R-hat: **{max_rhat:.4f}** (target < 1.01)\n")
        f.write(f"- Min ESS-bulk: **{min_ess:.0f}**\n\n")
        f.write("## Notes\n\n")
        f.write(
            "This pooled model serves only as a reference baseline. It cannot capture "
            "the city-level heterogeneity (~3× price gap between Warszawa and Częstochowa, "
            "see EDA §4) or the temporal dynamics (national +16% surge Aug–Dec 2023, "
            "EDA §5) that motivate Stages 2 and 3.\n"
        )


def main() -> None:
    print(f"Loading {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df = prepare_data(df)
    print(f"Fitting pooled baseline on {len(df):,} observations...")
    idata = fit_model(df)
    write_report(idata, n_obs=len(df))
    print(f"Saved report to: {REPORT_PATH}")
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    idata.to_netcdf(TRACE_PATH)
    print(f"Saved trace to: {TRACE_PATH}")


if __name__ == "__main__":
    main()
