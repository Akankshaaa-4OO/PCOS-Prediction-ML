# PCOS Prediction Using Machine Learning

A Jupyter notebook project that explores patient health and lifestyle data and compares machine learning models for predicting the recorded PCOS diagnosis label. It includes data exploration, preprocessing, class balancing with SMOTE, feature selection with Boruta, and model evaluation.

> **Important:** This is an educational machine learning project. It is not a medical device and must not be used to diagnose PCOS, guide treatment, or replace advice from a qualified healthcare professional.

## Project contents

```text
PCOS-Prediction-ML/
├── data/
│   └── last dataset.csv
├── notebook/
│   └── pcos_ML.ipynb
├── images/
├── model/
└── README.md
```

The notebook reads a labeled CSV dataset and uses `PCOS (Y/N)` as its target column. The included CSV contains 541 records and 29 columns (28 input columns and the target): 364 records labeled 0 and 177 labeled 1. The dataset includes a patient file number, demographic and health measurements, lifestyle indicators, and the PCOS label.

## Workflow

1. Explore the dataset and inspect missing values and correlations.
2. Split the data into training and test sets with stratification (80/20, random seed 42).
3. Handle missing values and scale the input features.
4. Apply SMOTE to the training split to address class imbalance.
5. Use Boruta with a Random Forest estimator for feature selection.
6. Train and evaluate Logistic Regression, Random Forest, and XGBoost models, including a tuned XGBoost configuration.

## Results recorded in the notebook

The saved notebook outputs show the following evaluation on a 109-row holdout set. The positive class is `PCOS (Y/N) = 1`; recall below refers to that class.

| Model | Accuracy | Positive-class recall |
|---|---:|---:|
| Logistic Regression | 87.16% | 0.89 |
| Random Forest | 87.16% | 0.81 |
| XGBoost | 85.32% | 0.86 |
| Tuned XGBoost | 85.32% | 0.86 |

These are the values in the notebook's saved evaluation outputs, not independently rerun results. The notebook's final comparison chart contains different values for Random Forest and tuned XGBoost, so treat the results as provisional until the notebook is rerun end to end and the chart is corrected. The notebook also reports 13 Boruta-selected features, while its model comparison manually uses 11 features and excludes `Patient File No.`.

## Requirements

Use Python 3.9 or newer and install the libraries imported by the notebook:

```bash
pip install jupyter pandas numpy matplotlib seaborn scikit-learn imbalanced-learn boruta xgboost
```

## Run the notebook

1. Clone or download this repository.
2. From the repository root, start Jupyter:

   ```bash
   jupyter notebook
   ```

3. Open `notebook/pcos_ML.ipynb` and run the cells in order.

### Data path note

The current notebook contains `pd.read_csv("  dataset.csv")`, which does not match the repository's data file path. Update that cell to point to the included dataset before running from the repository root:

```python
import pandas as pd

df = pd.read_csv("data/last dataset.csv")
```

The notebook processes the patient file number as an input during preprocessing and Boruta selection, although it later drops it from the manually selected model features. Consider excluding identifiers from model inputs and reviewing the preprocessing/evaluation pipeline before interpreting results.

## Data privacy

The CSV appears to contain patient-level records and a patient file number. Before publishing this repository publicly, confirm that you have permission to share the dataset and that it contains no identifiable or sensitive information. Remove or anonymize identifiers and any data that should not be public. Do not commit private patient data, credentials, or other confidential material.

## Disclaimer

Predictions from this project are experimental and may be inaccurate or biased. They are not clinical recommendations or a substitute for diagnosis by a healthcare professional. Do not use the model to make decisions about an individual's health.

