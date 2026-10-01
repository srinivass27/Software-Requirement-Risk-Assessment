import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, mean_squared_error, make_scorer

def get_scoring():
    return {
        'accuracy': 'accuracy',
        'auc': 'roc_auc_ovo',
        'f1': 'f1_macro',
        'precision': 'precision_macro',
        'recall': 'recall_macro',
        'error_rate': make_scorer(lambda y_true, y_pred: 1 - accuracy_score(y_true, y_pred)),
        'rmse': make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)))
    }

def evaluate_models(models, X, y, cv=5):
    fold_results = []
    scoring = get_scoring()

    for name, model in models.items():
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring)
        for fold_idx in range(cv):
            fold_results.append({
                'Model': name,
                'Fold': fold_idx + 1,
                'Accuracy': cv_results['test_accuracy'][fold_idx],
                'AUC': cv_results['test_auc'][fold_idx],
                'Error Rate': cv_results['test_error_rate'][fold_idx],
                'F1 Score': cv_results['test_f1'][fold_idx],
                'Precision': cv_results['test_precision'][fold_idx],
                'Recall': cv_results['test_recall'][fold_idx],
                'RMSE': cv_results['test_rmse'][fold_idx]
            })

    results_df = pd.DataFrame(fold_results)
    return results_df
