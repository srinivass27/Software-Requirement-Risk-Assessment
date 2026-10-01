from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def get_bagging_models():
    return {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42),
        "Bagging (DT)": BaggingClassifier(
            estimator=DecisionTreeClassifier(),
            n_estimators=50,
            random_state=42
        )
    }

def get_boosting_models():
    return {
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42),
        "LightGBM": LGBMClassifier(n_estimators=100, random_state=42),
    }
