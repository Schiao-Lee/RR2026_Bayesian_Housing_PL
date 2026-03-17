# Reproducible Bayesian Hierarchical Analysis of Apartment Prices in Poland 🇵🇱

**Course:** Reproducible Research UW - 2026  
**Instructor:** dr Jakub Michańków  

---

##  Team Members

| Name | Student ID | Role / Responsibilities | GitHub Profile |
| :--- | :--- | :--- | :--- |
| Levente Laszlo Szabo | 488761 | **Role name:**  | [@LeventeSzaboUW](https://github.com/LeventeSzaboUW) |
| Loveness Tafadzwa Mudyiwa | 478559 | **Role name:**  | [@username](https://github.com/) |
| Xiao Li | 473533 | **Role name:**  | [@Schiao-Lee](https://github.com/Schiao-Lee) |
| Szymon Grabowski  | 473037 | **Role name:** | [@username](https://github.com/) |
---

##  Project Overview

This project aims to identify the key determinants of real estate prices across the 15 largest Polish cities (Warsaw, Krakow, Wroclaw, etc.). Because real estate data naturally possesses a nested structure (apartments are grouped within cities), applying traditional linear regression may lead to either underfitting (complete pooling) or overfitting (no pooling).

To address this, we implement a **Bayesian Hierarchical Model (BHM)** using partial pooling. This approach allows each city to have its own baseline price (random intercepts) and specific price decay rates based on distance to the city center (random slopes), while still sharing statistical strength across the entire national dataset.

### Mathematical Formulation
At the core of our hierarchical model:
$$Price_i \sim \text{Normal}(\mu_i, \sigma)$$
$$\mu_i = \alpha_{city[i]} + \beta_{city[i]} \cdot centreDistance_i + \gamma \cdot squareMeters_i + \dots$$
Where:
* $\alpha_{city[i]}$ represents the varying baseline price for each city.
* $\beta_{city[i]}$ represents the varying effect of the distance to the city center for each city.

---

##  Dataset Information

* **Source:** [Apartment Prices in Poland (Kaggle)](https://www.kaggle.com/datasets/mlenkin/apartment-prices-in-poland)
* **Time Span:** August 2023 - June 2024 (Monthly snapshots).
* **Key Features:** `city`, `squareMeters`, `buildYear`, `centreDistance`, `poiCount` (OpenStreetMap POI data).
* **Reproducibility Note:** We **do not** store raw CSV files in this repository. All data is fetched programmatically via the Kaggle API to ensure a fully automated and reproducible data pipeline.

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

#  Repository Configuration & Git Workflow

To ensure the highest standards of code quality and reproducibility, this repository is configured with strict collaboration rules:

* **Protected `main` Branch:** Direct pushes to the `main` branch are strictly prohibited. 
* **Pull Request (PR) Requirement:** All new features, data pipelines, and modeling scripts must be developed on separate branches (e.g., `feature/data-cleaning`). They can only be merged into `main` via a Pull Request.
* **Peer Review:** Every PR requires at least one approval from another team member before it can be merged.
* **No Force Pushing:** Force pushing (`git push -f`) and history deletion are disabled to preserve the complete commit history for auditing by the course instructor.
* **Environment as Code:** All dependencies are strictly version-controlled via `environment.yml`. We use `conda` instead of `venv` to ensure stable compilation of underlying C++/Fortran libraries required for Bayesian inference (e.g., PyMC).
