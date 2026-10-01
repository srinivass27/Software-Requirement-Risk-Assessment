import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df = df.drop(columns=['Requirements'])

    # Encode categorical columns
    categorical_cols = df.select_dtypes(include='object').columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  

    df = df.dropna()
    X = df.drop(columns=['Risk Level'])
    y = df['Risk Level'] - 1
    return X, y, label_encoders

def balance_data(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled
