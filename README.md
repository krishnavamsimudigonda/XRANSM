# X-RANSM: A SMOTE-Enhanced Stacking Ensemble Model for Credit Card Fraud Detection

## Overview

**X-RANSM** is a robust hybrid machine learning framework designed to detect credit card fraud with high precision and recall. It utilizes a **stacked ensemble architecture** integrating Random Forest, XGBoost, and Artificial Neural Networks (ANN), with **SMOTE (Synthetic Minority Oversampling Technique)** applied to address class imbalance issues common in real-world datasets.

The model is validated on a real-world dataset with only **0.172%** fraudulent transactions and demonstrates superior performance across key metrics compared to individual models.

## Key Features

- Combats class imbalance using SMOTE
- Integrates Random Forest, XGBoost, and ANN in a stacking ensemble
- Implements soft voting for final classification
- Hyperparameter tuning using Grid Search and Bayesian Optimization
- Evaluated on metrics: Accuracy, Precision, Recall, F1-Score, and AUC-ROC

## Dataset

- **Source**: Kaggle - Credit Card Fraud Detection Dataset
- **Transactions**: 284,807
- **Fraudulent**: 492 (~0.172%)
- **Features**: 30 (PCA transformed for privacy)

## Model Architecture

1. **Data Preprocessing**: Z-score normalization for 'Time' and 'Amount' features.
2. **SMOTE Oversampling**: Generates synthetic examples of fraudulent transactions.
3. **Base Learners**:
   - **Random Forest (RF)**: Captures feature importance, reduces variance.
   - **XGBoost (XGB)**: Handles imbalance using gradient boosting with regularization.
   - **ANN**: Learns non-linear and temporal fraud patterns.
4. **Meta Learner**: Combines predictions using a soft voting mechanism.
5. **Evaluation**: Performance assessed via cross-validation and test data.

## Performance Summary

| Model            | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 97.80% | 88.52% | 79.41% | 83.71% | 0.941 |
| Random Forest       | 98.96% | 91.21% | 91.10% | 91.15% | 0.978 |
| XGBoost             | 99.11% | 92.00% | 92.45% | 92.22% | 0.985 |
| ANN                 | 98.73% | 90.18% | 89.34% | 89.76% | 0.972 |
| **X-RANSM**         | **99.43%** | **94.22%** | **93.78%** | **93.99%** | **0.991** |

## Requirements

- Python 3.9+
- scikit-learn
- imbalanced-learn
- xgboost
- tensorflow
- pandas, numpy, matplotlib, seaborn

Install dependencies using:

```bash
pip install -r requirements.txt
```

## How to Run

1. Clone this repository.
2. Run the `XRANSM Model.ipynb` Jupyter Notebook.
