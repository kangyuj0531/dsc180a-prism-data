# Beyond Credit Scores: Transaction-Level Modeling for Credit Risk

This repository contains the work for **DSC 180 HDSI Capstone with Prism Data**. The project develops behavior-based models that predict consumer credit risk using transaction-level and account-level signals. Data sources include application-level credit attributes, account balances, transaction cashflows, and spending category mappings.

---
## Project Website

Published via GitHub Pages (serve from the `docs/` folder): https://kangyuj0531.github.io/dsc180a-prism-data/


## Repository structure

- `feature_engineering/` — notebooks implementing feature construction and EDA (balance/time-series, cashflow, monthly features).
- `scoring_exclusions/` — notebooks and rules that define pre-modeling exclusion criteria (data-quality filters, insufficient-history rules); produces filtered datasets used downstream.
- `feature_selection/` — notebooks for automated and manual feature selection (forward/backward, RFE, mutual information, comprehensive selection).
- `models/` — modeling notebooks and experiments (`model_comparison.ipynb`, LightGBM/XGBoost/RF/RNN notebooks).
- `scripts/` — reusable Python modules for loading, backfilling, and feature creation (`data_loading.py`, `backfill_transactions.py`, `feature_creation.py`).
- `docs/` — static site assets for GitHub Pages (`index.html`, `style.css`, `script.js`).
- `Q1/` — prior quarter work and reference notebooks (EDA, preprocessing, income analyses).
- `README.md`, `requirements.txt`, `.gitignore`

---

## Data

The underlying raw datasets are restricted and are not included in this repository. These data cannot be publicly shared; analyses were performed on UCSD Datahub and local secure environments.

## Feature engineering

All feature engineering logic lives in `feature_engineering/` and `scripts/feature_creation.py`.

Highlights:

- Daily aggregation: helper `prepare_daily_data()` converts transaction/backfill rows into end-of-day series per consumer (end-of-day balance, label, net daily change, transaction counts).
- Windowed statistics: compute means/medians/min/max/std and simple linear trends over recent windows (7/30/90 days).
- Event counts & liquidity signals: counts of observed days, transaction counts, and short-term liquidity features.
- The pipeline generates ~250 feature candidates capturing balance dynamics, cashflow patterns, category-level spend, income characteristics, fees, and short-term risk.

- Scoring exclusions: rules and filters applied before modeling to remove accounts or consumers with insufficient history, data quality issues, or anomalous activity. The `scoring_exclusions/scoring_exclusions.ipynb` notebook documents the exclusion criteria and produces filtered datasets used downstream.

---

## Feature selection

The `feature_selection/` notebooks produce ranked and selected feature sets using:

- Forward / backward selection
- Recursive Feature Elimination (RFE)
- Mutual information selection
- Comprehensive selection and consensus-based feature lists

---

## Modeling

Model experiments are in `models/`. `models/model_comparison.ipynb` compares classifiers trained on selected feature sets (commonly top-50 features).

Typical models compared:

- Logistic Regression (`scikit-learn`)
- XGBoost (`xgboost`)
- LightGBM (`lightgbm`)
- Random Forest (`scikit-learn`)
- Gradient Boosting (`scikit-learn`)

Other modeling experiments and notebooks (not always included in the automated comparison):

- Decision Trees (`models/decision_tree.ipynb`)
- Standalone Logistic Regression (`models/logreg.ipynb`)
- RNN and sequence models (`models/rnn_model.ipynb`)
- XGBoost / LightGBM individual experiment notebooks (`models/xgboost.ipynb`, `models/lightgbm.ipynb`)
- Gradient-boosting standalone experiments (`models/gradient_boosting.ipynb`)

These notebooks contain additional architecture experiments and may include alternative evaluation pipelines or data preprocessing choices.

Evaluation setup:

- Metric: ROC-AUC (Train / Validation / Test)
- Split: 60 / 20 / 20 (stratified)

---

## Scripts

The `scripts/` package contains reusable code for programmatic runs:

- `data_loading.py` — utilities to load raw Parquet/CSV sources and produce standard DataFrames.
- `backfill_transactions.py` — fills missing historical transaction rows to produce continuous daily series per account.
- `feature_creation.py` — modular feature creation helpers (window stats, trends, aggregations).
- `feature_selection.py` — utilities for programmatic feature selection (forward/backward, RFE, mutual information).
- `model_data.py` — helpers for preparing model-ready datasets (train/val/test splits, label alignment).

These modules enable running the pipeline outside notebooks for reproducibility or batch processing.

---

## How to run

1. Environment: `python -m pip install -r requirements.txt` (or create a conda env then install).
2. Feature generation: run `feature_engineering/feature_creation.ipynb` or import helpers from `scripts.feature_creation`.
3. (Optional) Scoring exclusion: run `scoring_exclusions/scoring_exclusions.ipynb` to exclude consumers that does not qualify for scoring because of insufficient data.
4. Feature selection: run notebooks in `feature_selection/` to create ranked feature lists.
5. Model comparison: run `models/model_comparison.ipynb` to train and evaluate classifiers on selected features.

---

## Docs / Website

Static site assets in `docs/` are suitable for GitHub Pages. The site title is set in `docs/index.html`.
