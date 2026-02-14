# dsc180a-prism-data

This repository contains the Quarter 2 work for **DSC 180A: Prism Data**. The project focuses on  developing a behavior-based algorithm that a consumer's credit risk. We have information on the following data about the consumer: application-level credit attributes, account-level balance information, transaction-level cash flow activity, and spending category classifications.  

---

## Repository Structure

```
Q1/
    *files for Q1 work for reference

feature_engineering/
    01_balance_over_time.ipynb
    02_balance_over_time.ipynb
    balance_feature.ipynb
    cashflow.ipynb
    data_loading.ipynb
    feature_creation.ipynb
    monthly_cashflow.ipynb
    scoring_exclusions.ipynb

feature_selection/
    feature_selection.ipynb
    feature_selection_comprehensive.ipynb
    feature_selection_mutual_info.ipynb
    feature_selection_rfe.ipynb

models/
    decision_tree.ipynb
    lightgbm.ipynb
    randomforest.ipynb
    rnn_model.ipynb

scripts/
    __init__.py
    backfill_transactions.py
    data_loading.py
    feature_creation.py

README.md
.gitignore
requirements.txt
```
---

## 1. Feature Engineering

All feature engineering logic is located in **feature_engineering/**.

Includes:
- Consumer checking account balances over time  
- Balance and income related features  
- EDA on **good** and **bad** samples
- Creating model-ready datasets
- Scoring exclusions
 
After completing feature engineering, approximately 227 total features were generated, capturing patterns in balance dynamics, cashflow behavior, transaction activity, category-level spending, income characteristics, fee activity, and short-term liquidity risk.

---

## 2. Feature Selection

All feature selection occurs in **feature_selection/**.

Methodologies Include:
- Forward and backward selection  
- Comprehensive selection  
- Mutual info selection
- RFE selection

---

## 3. Models

Methodologies Include:
- Decision Trees
- LightGBM
- Random Forest
- Neural Networks
    -  RNNs

Each notebook includes accuracy, macro-F1, precision, confusion matrices, ROC-AUC, and latency comparisons.

---

## 4. Scripts

```
scripts/
    __init__.py
        Initializes the scripts package for module imports.

    backfill_transactions.py
        Fills missing historical transaction data ensures a time series of daily transactions for each consumer's checking account.

    data_loading.py
        Loads raw data sources and performs initial validation and formatting.

    feature_creation.py
        Generates engineered features from transaction and balance data.
```

---

## 5. Reproducibility

### Environment Setup
* `requirements.txt` includes CUDA-pinned PyTorch wheels; these install only on supported NVIDIA/CUDA systems. CPU-only machines (including Apple Silicon) will automatically receive CPU wheels.
* Conda users may create an environment first, then run: `python -m pip install -r requirements.txt`
* Quick one-liner (any environment): `python -m pip install -r requirements.txt`

## How to Run

1. Use **feature_engineering/feature_creation.ipynb** to generate cleaned datasets and features.
2. Run notebooks in **feature_selection/** to test different methods to create various feature sets.
3. Run notebooks in **models/** to train and evaluate specific classifiers.

---

## Notes

- All datasets found on and executed on UCSD Datahub and local laptop.
