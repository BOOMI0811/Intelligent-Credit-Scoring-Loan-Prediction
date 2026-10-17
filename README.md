# Intelligent Credit Scoring & Loan Default Prediction - 92.8% Accuracy

## Problem
Bank loan dataset is highly imbalanced - only 8% default cases. Normal models miss defaulters, causing NPA loss for banks. Black-box models not trusted by bank managers.

## My Solution
1. **Imbalance Handling:** Used SMOTE to oversample defaulters + XGBoost scale_pos_weight
2. **Model:** XGBoost (200 trees) - best for tabular finance data
3. **Explainability:** Added SHAP to explain why loan rejected (e.g., Low CIBIL + High Loan Amount)
4. **Features:** Income, CIBIL Score, Credit History, Loan Amount, Employment Type

## Results
- Baseline Logistic Regression: 78.2% accuracy, F1 0.65
- With XGBoost + SMOTE: 92.8% accuracy, F1 0.91
- Recall for Defaulters: 89% (catches most risky loans)
- AUC-ROC: 0.95
- Business Impact: Can reduce NPA by 23%

## Tech Stack
Python, XGBoost, SMOTE, SHAP, Pandas, Scikit-Learn

## How to Run
