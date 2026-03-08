# Credit Card Fraud Detection

## Overview

This repository contains an end-to-end notebook-based machine learning workflow for credit card fraud detection on a highly imbalanced transaction dataset. The project focuses on exploratory data analysis, feature engineering, imbalance handling, baseline model comparison, threshold optimization, and gradient boosting for improved fraud detection performance.

The implementation is centered around Jupyter notebooks and is intended for experimentation, analysis, and model comparison rather than packaged production deployment.

## Problem Statement

Credit card fraud detection is a binary classification problem with extreme class imbalance. In this project, the positive class represents fraudulent transactions and the negative class represents legitimate transactions. The main technical challenge is not raw accuracy, but detecting rare fraud cases while controlling false positives on normal transactions.

## Technical Objectives

This project is designed to:

- analyze the distribution and behavior of fraudulent versus legitimate transactions;
- engineer more model-friendly features from raw transaction attributes;
- mitigate class imbalance through undersampling and SMOTE oversampling;
- compare several baseline classifiers under the same validation setup;
- evaluate models using metrics suitable for imbalanced learning;
- improve decision quality through threshold tuning and hyperparameter search.

## Dataset

The dataset is stored locally at:

`data/creditcard.csv`

The workflow assumes the dataset contains:

- a binary target column named `Class`;
- transaction features including `Time` and `Amount`;
- anonymized PCA-like feature columns such as `V1` to `V28`.

Because fraud detection is highly imbalanced, the dataset is treated as an imbalanced classification problem throughout the project.

## Methodology

### 1. Exploratory Data Analysis

The notebooks inspect:

- schema and descriptive statistics;
- missing values;
- target-class distribution;
- univariate distributions for `Amount` and `Time`;
- class-conditional distributions for fraudulent and non-fraudulent transactions.

The analysis establishes that the target distribution is heavily skewed and that direct training on raw class proportions requires imbalance-aware modeling decisions.

### 2. Feature Engineering

The project transforms raw variables to improve learnability:

- `Time` is converted into an hour-of-day style feature to better represent cyclic transaction behavior;
- `Amount` is log-transformed to reduce extreme right skew;
- robust scaling is applied to transformed `Time` and `Amount` features.

Original `Time` and `Amount` columns are then removed to avoid redundant representations.

### 3. Train/Test Protocol

The dataset is split into training and test sets using stratified sampling. This is important because all resampling operations are applied only to the training split to avoid data leakage.

### 4. Imbalance Handling

Two strategies are explored:

- random undersampling for correlation analysis and baseline model experimentation;
- SMOTE oversampling for synthetic minority class expansion without discarding the majority class.

### 5. Outlier Reduction

The undersampled training data is analyzed using correlation heatmaps and boxplots. Outlier handling is then applied selectively using an IQR-based rule on highly informative features rather than globally across the entire feature set.

### 6. Representation Visualization

The notebooks use t-SNE to project the balanced training data into two dimensions for qualitative class-separation inspection.

### 7. Baseline Model Comparison

The following classifiers are evaluated on the undersampled training set using the same cross-validation protocol:

- Logistic Regression
- K-Nearest Neighbors
- Support Vector Classifier
- Random Forest

The baseline comparison is implemented with consistent settings and validation flow so model scores are directly comparable.

### 8. Fraud-Focused Evaluation

Model evaluation goes beyond validation accuracy and includes:

- confusion matrices;
- precision-recall analysis;
- average precision score;
- manual threshold tuning to trade off recall against false positive volume.

This is critical for fraud detection, where the operational cost of false positives and false negatives is asymmetric.

### 9. Advanced Modeling

After baseline experimentation, the workflow expands to:

- Logistic Regression with hyperparameter tuning via `GridSearchCV`;
- Logistic Regression trained on SMOTE-resampled data;
- XGBoost with imbalance-aware weighting via `scale_pos_weight`;
- XGBoost hyperparameter optimization via randomized search.

The later stages of the notebooks indicate that boosted tree models capture the non-linear fraud patterns more effectively than the linear baselines.

## Repository Structure

```text
cc_fraud_detect/
├── data/
│   └── creditcard.csv
├── notebooks/
│   └── scripts.ipynb
├── requirements.txt
└── README.md
```

## Dependencies

The project currently declares the following packages:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `scipy`
- `tensorflow`
- `imbalanced-learn`
- `jupyter`
- `xgboost`

## Environment Setup

### Option A: Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: POSIX Shell

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Running the Project

This project is notebook-driven. After installing the dependencies, start Jupyter:

```bash
jupyter notebook
```

## Modeling Workflow Summary

The implemented pipeline can be summarized as follows:

1. Load transaction data.
2. Inspect schema, summary statistics, and class imbalance.
3. Visualize univariate and bivariate behavior of the target and key features.
4. Engineer scaled time and amount features.
5. Split data using stratified train/test sampling.
6. Build an undersampled training set for interpretability and baseline experiments.
7. Analyze correlations and remove selected outliers.
8. Compare four baseline classifiers with the same validation procedure.
9. Evaluate the best candidates on the original imbalanced test set.
10. Tune thresholds and optimize hyperparameters.
11. Compare undersampling, SMOTE, and XGBoost-based approaches.
12. Inspect remaining false negatives for error analysis.

## Evaluation Strategy

This repository emphasizes metrics that are meaningful for imbalanced binary classification:

- cross-validation score for controlled baseline comparison;
- confusion matrix for operational interpretation;
- precision and recall for fraud-capture tradeoffs;
- average precision / area under the precision-recall curve for ranking quality.

Accuracy alone is intentionally not treated as the primary success criterion, because it is not informative under extreme imbalance.

## Current Scope and Limitations

This repository is best described as a research and experimentation workspace. It currently does not include:

- a standalone training script or CLI entry point;
- automated tests;
- dataset download automation;
- model serialization and versioning;
- inference service or deployment artifacts.

The current implementation is therefore suitable for analysis and reproducible notebook experimentation, but not yet for direct production deployment.

## Reproducibility Notes

- Several steps use fixed random seeds such as `random_state=42` for comparability.
- Results may still vary slightly depending on library versions and notebook execution order.
- If you rerun experiments from scratch, execute cells sequentially because later steps depend on variables created earlier in the notebooks.

