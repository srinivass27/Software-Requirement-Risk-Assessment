# Software-Requirement-Risk-Assessment
Ensemble machine learning project for software requirement risk assessment using bagging and boosting algorithms.
# Ensemble Learning for Software Requirement Risk Assessment

## Overview

This project uses ensemble machine learning techniques to assess
risk levels associated with software requirements.

## Objective

The objective is to classify software requirements according to
their associated risk level using multiple ensemble learning models.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- Imbalanced-learn
- Matplotlib
- Seaborn

## Machine Learning Models

- Random Forest
- Extra Trees
- Bagging with Decision Trees
- Gradient Boosting
- XGBoost
- LightGBM

## Workflow

Data
↓
Preprocessing
↓
Categorical Encoding
↓
SMOTE
↓
Ensemble Models
↓
5-Fold Cross Validation
↓
Performance Evaluation

## Evaluation Metrics

- Accuracy
- AUC
- F1 Score
- Precision
- Recall
- Error Rate
- RMSE

## Results

The model evaluation results are available in:

`results/model_performance.csv`

Visualizations are available in:

`results/figures/`

## Project Structure

...
 
## How to Run

```bash
pip install -r requirements.txt
python main.py
