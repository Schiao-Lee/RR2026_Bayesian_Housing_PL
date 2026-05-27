"""
Script: 03_hierarchical_tvp.py
Stage 3 — Bayesian Hierarchical Time-Varying Parameters (TVP) Model.

Implements the full README state-space specification with random walks on
BOTH the city intercept and the city-specific centreDistance slope:

    log_price_{i,c,t} ~ Normal(μ_{i,c,t}, σ)
    μ_{i,c,t} = α_{c,t} + β_{c,t} · dist_z + γ · sqm_z + δ · poi_z

    α_{c,t} = α_{c,t-1} + η^α_{c,t},   η^α ~ Normal(0, σ_α)
    β_{c,t} = β_{c,t-1} + η^β_{c,t},   η^β ~ Normal(0, σ_β)
    α_{c,0} ~ Normal(ᾱ, τ_α)
    β_{c,0} ~ Normal(β̄, τ_β)

Parameterization choices for sampler stability:
- All random walks use non-centered form: innovations are N(0,1), scaled by
  the global walk variance σ_α / σ_β. This is the standard remedy for the
  centered-funnel pathology that has historically blocked hierarchical TVP
  models from converging under NUTS.
- Tight HalfNormal priors on σ_α (0.05) and σ_β (0.02) reflect the empirical
  observation from EDA §5 that the national price drift between adjacent
  monthly snapshots is small relative to cross-city heterogeneity.

Data: stratified ~20% subsample by city × snapshot_month (seed=42). The
smallest cell (Rzeszów × any month) retains ~25 observations — comfortably
enough to identify a single city × month state. Full-data re-fit is a
one-line change when the final write-up is prepared.

Knobs (env vars; defaults run the real overnight model):
    STAGE3_DRAWS, STAGE3_TUNE, STAGE3_CHAINS, STAGE3_TARGET_ACCEPT
"""

import os
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "master_sales_dataset.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "05_stage3_tvp_summary.md"
TRACE_PATH = PROJECT_ROOT / "data" / "traces" / "stage3_tvp.nc"
INTERCEPT_FIG_PATH = (
    PROJECT_ROOT / "notebooks" / "figures" / "stage3_city_intercept_trajectories.png"
)
SLOPE_FIG_PATH = (
    PROJECT_ROOT / "notebooks" / "figures" / "stage3_city_slope_trajectories.png"
)

FEATURES = ["squareMeters", "centreDistance", "poiCount"]
SAMPLE_FRAC = 0.20
SAMPLE_SEED = 42

DRAWS = int(os.environ.get("STAGE3_DRAWS", 1000))
TUNE = int(os.environ.get("STAGE3_TUNE", 1500))
CHAINS = int(os.environ.get("STAGE3_CHAINS", 2))
TARGET_ACCEPT = float(os.environ.get("STAGE3_TARGET_ACCEPT", 0.9))


def prepare_data(df: pd.DataFrame):
    df = df.dropna(subset=["price", "city", "snapshot_month", *FEATURES]).copy()
    df = (
        df.groupby(["city", "snapshot_month"], group_keys=False, observed=True)
        .sample(frac=SAMPLE_FRAC, random_state=SAMPLE_SEED)
        .reset_index(drop=True)
    )
    df["log_price"] = np.log(df["price"])
    for col in FEATURES:
        df[f"{col}_z"] = (df[col] - df[col].mean()) / df[col].std()

    cities = sorted(df["city"].unique())
    months = sorted(df["snapshot_month"].unique())
    df["city_idx"] = df["city"].map({c: i for i, c in enumerate(cities)})
    df["month_idx"] = df["snapshot_month"].map({m: i for i, m in enumerate(months)})
    return df, cities, months


def build_and_fit(df: pd.DataFrame, cities, months) -> az.InferenceData:
    n_cities = len(cities)
    n_months = len(months)
    coords = {"city": cities, "month": months, "obs": np.arange(len(df))}
    city_idx_np = df["city_idx"].to_numpy()
    month_idx_np = df["month_idx"].to_numpy()
    dist_z_np = df["centreDistance_z"].to_numpy()
    sqm_z_np = df["squareMeters_z"].to_numpy()
    poi_z_np = df["poiCount_z"].to_numpy()
    y_np = df["log_price"].to_numpy()

    with pm.Model(coords=coords):
        # Hyperpriors --------------------------------------------------------
        grand_alpha = pm.Normal("grand_alpha", mu=13.3, sigma=1.0)
        tau_alpha = pm.HalfNormal("tau_alpha", sigma=1.0)
        grand_beta = pm.Normal("grand_beta", mu=0.0, sigma=0.5)
        tau_beta = pm.HalfNormal("tau_beta", sigma=0.5)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=0.05)
        sigma_beta = pm.HalfNormal("sigma_beta", sigma=0.02)
        sigma = pm.HalfNormal("sigma", sigma=0.5)

        # Initial states (non-centered) -------------------------------------
        alpha0_z = pm.Normal("alpha0_z", mu=0.0, sigma=1.0, dims="city")
        beta0_z = pm.Normal("beta0_z", mu=0.0, sigma=1.0, dims="city")
        alpha0 = grand_alpha + tau_alpha * alpha0_z
        beta0 = grand_beta + tau_beta * beta0_z

        # Random walk innovations (non-centered) ----------------------------
        alpha_innov_z = pm.Normal(
            "alpha_innov_z", mu=0.0, sigma=1.0, shape=(n_cities, n_months - 1)
        )
        beta_innov_z = pm.Normal(
            "beta_innov_z", mu=0.0, sigma=1.0, shape=(n_cities, n_months - 1)
        )

        # Cumulative walk increments anchored at zero at t=0
        alpha_walk = pt.concatenate(
            [pt.zeros((n_cities, 1)), pt.cumsum(sigma_alpha * alpha_innov_z, axis=1)],
            axis=1,
        )
        beta_walk = pt.concatenate(
            [pt.zeros((n_cities, 1)), pt.cumsum(sigma_beta * beta_innov_z, axis=1)],
            axis=1,
        )

        alpha_ct = pm.Deterministic(
            "alpha_ct", alpha0[:, None] + alpha_walk, dims=("city", "month")
        )
        beta_ct = pm.Deterministic(
            "beta_ct", beta0[:, None] + beta_walk, dims=("city", "month")
        )

        # Time-invariant pooled slopes for the remaining features -----------
        gamma_sqm = pm.Normal("gamma_sqm", mu=0.0, sigma=0.5)
        delta_poi = pm.Normal("delta_poi", mu=0.0, sigma=0.5)

        mu = (
            alpha_ct[city_idx_np, month_idx_np]
            + beta_ct[city_idx_np, month_idx_np] * dist_z_np
            + gamma_sqm * sqm_z_np
            + delta_poi * poi_z_np
        )
        pm.Normal("y", mu=mu, sigma=sigma, observed=y_np, dims="obs")

        print(
            f"[{pd.Timestamp.now():%H:%M:%S}] Starting NUTS: "
            f"chains={CHAINS}, draws={DRAWS}, tune={TUNE}, "
            f"target_accept={TARGET_ACCEPT}"
        )
        idata = pm.sample(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            target_accept=TARGET_ACCEPT,
            random_seed=42,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )
    return idata


def write_outputs(idata: az.InferenceData, df: pd.DataFrame, cities, months) -> None:
    # 1. Persist trace first so any later plotting failure doesn't lose work.
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    idata.to_netcdf(TRACE_PATH)
    print(f"[{pd.Timestamp.now():%H:%M:%S}] Saved trace: {TRACE_PATH}")

    alpha_mean = idata.posterior["alpha_ct"].mean(("chain", "draw"))
    beta_mean = idata.posterior["beta_ct"].mean(("chain", "draw"))

    # 2. Trajectory plots ----------------------------------------------------
    INTERCEPT_FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(months))

    fig, ax = plt.subplots(figsize=(12, 7))
    for ci, city in enumerate(cities):
        ax.plot(x, alpha_mean[ci, :], marker="o", label=city, lw=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45)
    ax.set_ylabel(r"Posterior mean $\alpha_{c,t}$ (log PLN)")
    ax.set_title("Stage 3 — City-specific time-varying intercepts")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(INTERCEPT_FIG_PATH, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 7))
    for ci, city in enumerate(cities):
        ax.plot(x, beta_mean[ci, :], marker="o", label=city, lw=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45)
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_ylabel(r"Posterior mean $\beta_{c,t}$ (centreDistance slope)")
    ax.set_title("Stage 3 — City-specific time-varying distance sensitivity")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(SLOPE_FIG_PATH, dpi=150)
    plt.close(fig)
    print(f"[{pd.Timestamp.now():%H:%M:%S}] Saved figures.")

    # 3. Markdown report -----------------------------------------------------
    pop_vars = [
        "grand_alpha", "grand_beta", "gamma_sqm", "delta_poi",
        "sigma", "sigma_alpha", "sigma_beta", "tau_alpha", "tau_beta",
    ]
    pop_summary = az.summary(idata, var_names=pop_vars, round_to=4)
    full_summary = az.summary(idata, round_to=4)

    n_div = int(idata.sample_stats["diverging"].sum())
    n_draws = int(idata.sample_stats.sizes["draw"] * idata.sample_stats.sizes["chain"])
    max_rhat = float(full_summary["r_hat"].max())
    min_ess = float(full_summary["ess_bulk"].min())

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("# Stage 3 — Bayesian Hierarchical Time-Varying Parameters (TVP)\n\n")
        f.write(
            f"**Observations:** {len(df):,} "
            f"(stratified {SAMPLE_FRAC:.0%} subsample by city × snapshot_month, "
            f"seed={SAMPLE_SEED}; {len(cities)} cities × {len(months)} months)\n\n"
        )
        f.write(
            "**Model:** State-space TVP. Both `α_{c,t}` (city intercept) and "
            "`β_{c,t}` (city-specific centreDistance slope) follow non-centered "
            "Gaussian random walks across the 11 monthly snapshots. "
            "`squareMeters` and `poiCount` slopes are pooled (time- and "
            "city-invariant) to keep the parameter count tractable. Gaussian likelihood.\n\n"
        )
        f.write(
            f"**Sampler:** NUTS via PyMC — {CHAINS} chains × {DRAWS} draws "
            f"({TUNE} tune), target_accept={TARGET_ACCEPT}, seed=42\n\n"
        )

        f.write("## Population / hyperparameter posterior\n\n```\n")
        f.write(pop_summary.to_string())
        f.write("\n```\n\n")

        # City trajectory change table
        idx_early = list(range(min(3, len(months))))
        idx_late = list(range(max(0, len(months) - 3), len(months)))
        f.write(
            f"## City intercept change "
            f"(α: mean over months {idx_early} vs {idx_late})\n\n"
        )
        f.write("| City | α early | α late | Δ log PLN | Δ % (≈) |\n|---|---:|---:|---:|---:|\n")
        rows = []
        for ci, city in enumerate(cities):
            early = float(alpha_mean[ci, idx_early].mean())
            late = float(alpha_mean[ci, idx_late].mean())
            rows.append((city, early, late, late - early))
        for city, early, late, delta in sorted(rows, key=lambda r: -r[3]):
            f.write(
                f"| {city} | {early:+.3f} | {late:+.3f} | {delta:+.3f} | "
                f"{100 * (np.exp(delta) - 1):+.1f}% |\n"
            )
        f.write("\n")

        f.write("## Convergence\n\n")
        f.write(f"- Max R-hat (overall): **{max_rhat:.4f}** (target < 1.01)\n")
        f.write(f"- Min ESS-bulk (overall): **{min_ess:.0f}**\n")
        f.write(
            f"- Divergences after tuning: **{n_div}** / {n_draws} draws "
            f"({100 * n_div / n_draws:.2f}%)\n\n"
        )

        f.write("## Figures\n\n")
        f.write(
            f"- City intercept trajectories: "
            f"`{INTERCEPT_FIG_PATH.relative_to(PROJECT_ROOT)}`\n"
        )
        f.write(
            f"- City distance-slope trajectories: "
            f"`{SLOPE_FIG_PATH.relative_to(PROJECT_ROOT)}`\n\n"
        )

        f.write("## Comparison anchor\n\n")
        f.write(
            "EDA §5 documented a +16% national median price surge between "
            "Aug 2023 and Dec 2023, followed by a plateau through Jun 2024. "
            "The α_{c,t} trajectory plot should reproduce that as a roughly "
            "common drift across cities, with the divergences noted in EDA §5 "
            "(Kraków/Gdańsk strong appreciation; Katowice early-2024 decline; "
            "Białystok small-N noise). Sign-flip context: the static Stage 2 "
            "population centreDistance slope was −0.057; β_{c,t} should "
            "fluctuate around that level once city-time variation is allowed.\n"
        )

    print(f"[{pd.Timestamp.now():%H:%M:%S}] Saved report: {REPORT_PATH}")


def main() -> None:
    print(f"[{pd.Timestamp.now():%H:%M:%S}] Loading {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df, cities, months = prepare_data(df)
    print(
        f"[{pd.Timestamp.now():%H:%M:%S}] "
        f"Prepared {len(df):,} obs / {len(cities)} cities × {len(months)} months. "
        f"Per-cell size: min={df.groupby(['city','snapshot_month']).size().min()}, "
        f"median={int(df.groupby(['city','snapshot_month']).size().median())}, "
        f"max={df.groupby(['city','snapshot_month']).size().max()}"
    )
    idata = build_and_fit(df, cities, months)
    write_outputs(idata, df, cities, months)
    print(f"[{pd.Timestamp.now():%H:%M:%S}] Done.")


if __name__ == "__main__":
    main()
