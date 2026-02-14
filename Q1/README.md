# dsc180a-prism-data

This repository contains the Quarter 1 work for **DSC 180A: Prism Data**. The project focuses on cleaning consumer-transaction data, engineering memo and financial features, building multiple classification models, and analyzing income patterns.

---

## Repository Structure

```
eda/
    overview_eda.ipynb

income_prediction/
    income_eda.ipynb

preprocessing/
    feature_engineering.ipynb
    memos_parsed.ipynb

models/
    baseline_models.ipynb
    hist_gradient_boosting_classifier.ipynb
    neural_network_experiments.ipynb
    tiny_transformer_classifier.ipynb
    distilbert_classifier.ipynb
    huggingface_transformers.ipynb

README.md
.gitignore
requirements.txt
```
---

## 1. Data Cleaning and Preprocessing

All preprocessing logic is located in **preprocessing/**.

Includes:
- Parsing and standardizing transaction memos  
- Cleaning dates, numeric fields, and rare categories  
- Engineering temporal and behavioral features  
- Constructing reusable featurizers  
- Producing model-ready datasets  

---

## 2. Classification Models

All model development appears in **models/**.

### Baseline Models
- Logistic Regression  
- Random Forest Classifier
- XGBoost Classifier  

### Tree-Based Model
- **HistGradientBoostingClassifier**  
  - TF-IDF → SVD memo embeddings paired with boosted trees  

### Neural and Transformer Models
- Feed-Forward MLP  
- Tiny Transformer classifier  
- DistilBERT (Fine-tuned for sequence classification)
- RoBERTa (Fine-tuned)

Each notebook includes accuracy, macro-F1, confusion matrices, and latency comparisons.

---

## 3. Income EDA and Regularity Detection

Implemented in **income_prediction/income_eda.ipynb**.

### Consumer-Level Statistics
- Transaction counts  
- Inflow totals and variability  
- Active transaction span  
- Income grouping via percentiles  

### Major Income Sources
- PAYCHECK  
- INVESTMENT_INCOME  
- OTHER_BENEFITS  
- UNEMPLOYMENT_BENEFITS  

### Regular Income Detection
Recurring cycles (weekly, biweekly, monthly, every *N* days) identified using:
- Amount clustering  
- Interval analysis  
- Stability thresholds  

**Results**
- 1318 regular income patterns detected  
- 1046 consumers show consistent cycles  

## Key Questions

### What counts as income?
PAYCHECK, BENEFITS, and INVESTMENT_INCOME; excludes transfers, refunds, loans, and cash deposits.

### Must income be regular?
No. Many valid income sources (benefits, investments) are irregular.

### How is regularity detected?
By clustering similar inflows and analyzing gaps between payment dates.

---
## 4. Reproducibility

### Environment Setup
* `requirements.txt` includes CUDA-pinned PyTorch wheels; these install only on supported NVIDIA/CUDA systems. CPU-only machines (including Apple Silicon) will automatically receive CPU wheels.
* Conda users may create an environment first, then run: `python -m pip install -r requirements.txt`
* Quick one-liner (any environment): `python -m pip install -r requirements.txt`

## How to Run

1. Use **feature_engineering.ipynb** to generate cleaned datasets and features.  
2. Run notebooks in **models/** to train and evaluate specific classifiers.  
3. Use **income_prediction/income_eda.ipynb** for income-pattern analysis.
---

## Notes

- All work executed on UCSD Datahub and local laptop.  
- Transformer models require more memory and runtime.  
- Logistic Regression provides the strongest classical baseline.
