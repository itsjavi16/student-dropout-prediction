"""
Preprocessing pipeline: stratified split, scaling, one-hot encoding.

Column groups are defined here once so every notebook (EDA, modeling,
GPA trajectory) uses the same definitions.
"""

from os import path
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler



RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "dataset.csv"
target_col = "target"


# Nominal categorical columns: coded as intergers in the raw file
# but there is no ordinal relationship between the values (e.g Course, Nationality, etc.)
# One-hot encoding is appropriate for these columns.
CATEGORICAL_COLS = [
    "Marital status",
    "Application mode",
    "Course",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
]

# Binary categorical columns: coded as 0/1 in the raw file, so no need to one-hot encode.
BINARY_COLS = [
    "Daytime/evening attendance",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]

# True numeric columns: these are continuous variables that can be scaled for Logistic Regression.
NUMERIC_COLS = [
    "Application order",
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]

def load_data(path: Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)

def stratified_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    x = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(x, y, test_size=test_size, stratify=y, random_state=random_state)

def build_preprocessor() -> ColumnTransformer:
    """
    ColumnTransformer for Logistic Regression:
    - numeric columns -> StandardScaler
    - categorical columns -> OneHotEncoder
    - binary columns -> passthrough (already 0/1)

    Fit this on X_train only, then transform() both X_train and X_test to
    avoid leaking test-set information into the scaling/encoding.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
            ("bin", "passthrough", BINARY_COLS),
        ]
    )

def build_preprocessing_pipeline() -> Pipeline:
    """
    Returns a preprocessing pipeline that applies the ColumnTransformer to the data.
    """
    preprocessor = build_preprocessor()
    return Pipeline(steps=[("preprocessor", preprocessor)])

if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = stratified_split(df)
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
    print("Train class balance:\n", y_train.value_counts(normalize=True).round(3))
    print("Test class balance:\n", y_test.value_counts(normalize=True).round(3))