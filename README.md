# Reproducible Bayesian Hierarchical Time-Varying Analysis of Apartment Prices in Poland 🇵🇱

[![CI](https://github.com/Schiao-Lee/RR2026_Bayesian_Housing_PL/actions/workflows/ci.yml/badge.svg)](https://github.com/Schiao-Lee/RR2026_Bayesian_Housing_PL/actions/workflows/ci.yml)
[![Pages](https://github.com/Schiao-Lee/RR2026_Bayesian_Housing_PL/actions/workflows/pages.yml/badge.svg)](https://github.com/Schiao-Lee/RR2026_Bayesian_Housing_PL/actions/workflows/pages.yml)

📄 **Read the rendered reports without cloning:** <https://schiao-lee.github.io/RR2026_Bayesian_Housing_PL/>

**Courses:**  
- Reproducible Research UW - 2026 (dr Jakub Michańków)   

---

##  Team Members

Roles below are primary, not exclusive — members may contribute across areas as the work demands.

| Name | Student ID | Role / Responsibilities | GitHub Profile |
| :--- | :--- | :--- | :--- |
| Levente Laszlo Szabo | 488761 | Data pipeline, EDA, visualizations | [@LeventeSzaboUW](https://github.com/LeventeSzaboUW) |
| Xiao Li | 473533 | Modeling (Pooled → BHM → TVP state-space) | [@Schiao-Lee](https://github.com/Schiao-Lee) |
| Szymon Grabowski | 473037 | Report, documentation | [@sgrabowski8](https://github.com/sgrabowski8) |

---

##  Project Overview

This project aims to identify the key determinants of real estate prices across the 15 largest Polish cities (Warsaw, Krakow, Wroclaw, etc.) **and to track how these determinants evolve over time**. Because real estate data naturally possesses a nested structure (apartments are grouped within cities) and a temporal dimension (monthly snapshots from August 2023 to June 2024), applying traditional linear regression may lead to either underfitting (complete pooling) or overfitting (no pooling), while also failing to capture the dynamics of a changing housing market.

To address this, we implement a **Bayesian Hierarchical Time-Varying Parameters (TVP) Model** using partial pooling and state-space formulations. This approach allows each city to have its own baseline price and price-sensitivity coefficients that **evolve over time as random walks**, while still sharing statistical strength across the entire national dataset. By combining the hierarchical structure with time-varying dynamics, the model simultaneously serves two purposes:

1. **Reproducible Research:** A fully automated, end-to-end reproducible pipeline — from data acquisition to posterior inference — meeting strict reproducibility standards.
2. **Bayesian Time-Series Econometrics:** A state-space / TVP model estimated via MCMC, directly aligned with the course syllabus (TVP, state-space models, Markov Chain Monte Carlo).

### Mathematical Formulation

#### Observation Equation

At each time period $t$ and for each apartment $i$ in city $c$:

$$Price_{i,c,t} \sim \text{Normal}(\mu_{i,c,t},\ \sigma^2)$$

$$\mu_{i,c,t} = \alpha_{c,t} + \beta_{c,t} \cdot centreDistance_{i} + \gamma_t \cdot squareMeters_{i} + \dots$$

Where:
- $\alpha_{c,t}$ is the **time-varying** baseline price for city $c$ at month $t$.
- $\beta_{c,t}$ is the **time-varying** effect of distance to city center for city $c$ at month $t$.
- $\gamma_t$ is the (optionally time-varying) effect of apartment size.

#### Transition Equations (State Evolution)

The key time-series component: city-level parameters follow **random walk** processes, capturing the gradual evolution of the housing market:

$$\alpha_{c,t} = \alpha_{c,t-1} + \eta_{c,t}^{\alpha}, \quad \eta_{c,t}^{\alpha} \sim \text{Normal}(0,\ \sigma_{\alpha}^2)$$

$$\beta_{c,t} = \beta_{c,t-1} + \eta_{c,t}^{\beta}, \quad \eta_{c,t}^{\beta} \sim \text{Normal}(0,\ \sigma_{\beta}^2)$$

#### Hierarchical Priors

The initial states are drawn from city-group-level distributions (partial pooling):

$$\alpha_{c,0} \sim \text{Normal}(\bar{\alpha},\ \tau_{\alpha}^2)$$

$$\beta_{c,0} \sim \text{Normal}(\bar{\beta},\ \tau_{\beta}^2)$$

Hyperpriors:

$$\bar{\alpha} \sim \text{Normal}(0,\ 10^2), \quad \tau_{\alpha} \sim \text{HalfNormal}(s_{\alpha})$$

$$\sigma_{\alpha},\ \sigma_{\beta} \sim \text{HalfNormal}(s)$$

This structure forms a **state-space model**: the observation equation links observed prices to latent time-varying parameters, and the transition equations govern how those parameters evolve — a canonical framework in Bayesian time-series econometrics.

### Modeling Strategy

We adopt a progressive modeling approach for comparison and validation:

| Stage | Model | Purpose |
| :--- | :--- | :--- |
| 1 | Pooled Bayesian Linear Regression | Baseline — ignores city and time structure |
| 2 | Bayesian Hierarchical Model (BHM) | Adds city-level partial pooling (random intercepts & slopes) |
| 3 | Bayesian Hierarchical TVP Model | Full model — adds time-varying parameters via state-space |

Model comparison will be conducted using **WAIC / LOO-CV** (via ArviZ), allowing us to quantify the empirical gain from incorporating temporal dynamics.

---

##  Dataset Information

- **Source:** [Apartment Prices in Poland (Kaggle)](https://www.kaggle.com/datasets/krzysztofjamroz/apartment-prices-in-poland)
- **Time Span:** August 2023 – June 2024 (11 monthly snapshots, providing $T = 11$ time periods)
- **Key Features:** `city`, `squareMeters`, `buildYear`, `centreDistance`, `poiCount` (OpenStreetMap POI data)
- **Temporal Structure:** Each monthly CSV is treated as a cross-sectional snapshot at time $t$, enabling the construction of a panel dataset indexed by $(city, month)$.
- **Reproducibility Note:** We **do not** store raw CSV files in this repository. All data is fetched programmatically via the Kaggle API to ensure a fully automated and reproducible data pipeline.

---

##  Reproducibility & Workflow

To meet the strict reproducibility standards of this course, our repository is designed to be fully automated.

### 1. Environment Setup

We ensure dependency consistency using Conda. To recreate our exact computational environment:
```bash
# Clone the repository
git clone git@github.com:Schiao-Lee/RR2026_Bayesian_Housing_PL.git
cd RR2026_Bayesian_Housing_PL

# Create and activate the Conda environment
conda env create -f environment.yml
conda activate bayes_housing_env
```

### Repository Configuration & Git Workflow

To ensure the highest standards of code quality and reproducibility, this repository is configured with strict collaboration rules:

- **Protected `main` Branch:** Direct pushes to the `main` branch are strictly prohibited.
- **Pull Request (PR) Requirement:** All new features, data pipelines, and modeling scripts must be developed on separate branches (e.g., `feature/data-cleaning`). They can only be merged into `main` via a Pull Request.
- **Peer Review:** Every PR requires at least one approval from another team member before it can be merged.
- **No Force Pushing:** Force pushing (`git push -f`) and history deletion are disabled to preserve the complete commit history for auditing by the course instructor.
- **Environment as Code:** All dependencies are strictly version-controlled via `environment.yml`. We use `conda` instead of `venv` to ensure stable compilation of underlying C++/Fortran libraries required for Bayesian inference (e.g., PyMC).

### 2. Automated Data Acquisition

Ensure you have your Kaggle API credentials (`kaggle.json`) configured. Run the following command to automatically download, unzip, and merge the monthly datasets:
```bash
python src/data_prep/01_download_and_merge.py
```

### 3. Exploratory Data Analysis

```bash
python src/analysis/02_eda.py
# or open the notebook
jupyter notebook notebooks/02_eda.ipynb
```

### 4. Model Estimation

```bash
# Stage 1: Pooled baseline
python src/models/01_pooled_baseline.py

# Stage 2: Hierarchical BHM (partial pooling, no time dynamics)
python src/models/02_hierarchical_bhm.py

# Stage 3: Hierarchical TVP (full time-varying model)
python src/models/03_hierarchical_tvp.py
```

### 5. Model Comparison & Diagnostics

```bash
# WAIC / LOO-CV comparison, trace plots, posterior summaries
python src/analysis/model_comparison.py
```

---

##  Results

| Output | Path |
| :--- | :--- |
| EDA findings (priors, feature selection) | [`reports/02_eda_findings.md`](reports/02_eda_findings.md) |
| Stage 1 — Pooled baseline | [`reports/03_stage1_pooled_summary.md`](reports/03_stage1_pooled_summary.md) |
| Stage 2 — Hierarchical BHM | [`reports/04_stage2_hierarchical_summary.md`](reports/04_stage2_hierarchical_summary.md) |
| Stage 3 — TVP (state-space) | [`reports/05_stage3_tvp_summary.md`](reports/05_stage3_tvp_summary.md) |
| Model comparison (LOO-CV / WAIC) | [`reports/06_model_comparison.md`](reports/06_model_comparison.md) |
| City-specific time-varying intercepts plot | [`notebooks/figures/stage3_city_intercept_trajectories.png`](notebooks/figures/stage3_city_intercept_trajectories.png) |
| City-specific time-varying distance slopes plot | [`notebooks/figures/stage3_city_slope_trajectories.png`](notebooks/figures/stage3_city_slope_trajectories.png) |

**Headline:** Stage 3 (TVP) wins LOO-CV with **98.3% stacking weight**; elpd gains are **+18,768 (Stage 1 → 2)** and **+2,302 (Stage 2 → 3)** on the common 39k-row subsample. Both hierarchy and time variation pay their way under cross-validation. See `reports/06_model_comparison.md` for the full table.
