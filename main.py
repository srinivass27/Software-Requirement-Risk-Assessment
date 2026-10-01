import os
from src.data_preprocessing import load_and_preprocess_data, balance_data
from src.visualization import plot_class_distribution
from src.models import get_bagging_models, get_boosting_models
from src.evaluation import evaluate_models

# Get path relative to project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "data", "output.csv")

# Load and preprocess
X, y, label_encoders = load_and_preprocess_data(csv_path)

# Plot original distribution
plot_class_distribution(y, "Original Class Distribution", save_path=os.path.join(BASE_DIR, "results/figures/original_distribution.png"))

# Balance data
X_resampled, y_resampled = balance_data(X, y)

# Plot balanced distribution
plot_class_distribution(y_resampled, "Balanced Class Distribution", save_path=os.path.join(BASE_DIR, "results/figures/balanced_distribution.png"))

# Get models
models = {**get_bagging_models(), **get_boosting_models()}

# Evaluate
results_df = evaluate_models(models, X_resampled, y_resampled)

# Save results
results_df.to_csv(os.path.join(BASE_DIR, "results/model_performance.csv"), index=False)
print("\n=== Cross-Validation Results ===")
print(results_df.groupby("Model").mean()[["Accuracy", "AUC", "F1 Score"]])
