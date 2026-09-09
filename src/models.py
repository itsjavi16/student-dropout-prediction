"""
Model definitions for comparing interpretability vs. accuracy:
Logistic Regression, Random Forest, XGBoost.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


def get_logistic_regression(random_state=42) -> LogisticRegression:
    """
    Returns a Logistic Regression model with class_weights = "balanced".
    Reweights the loss inversely proportional to class frequencies, so erros such as
    "Enrolled" count more than errors on the more common class "Graduate" class    
    
    """
    return LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=random_state)

def get_random_forest(random_state=42) -> RandomForestClassifier:
    """
    Returns a Random Forest model with class_weights = "balanced".
    """
    return RandomForestClassifier( 
        n_estimators=300, 
        class_weight="balanced", 
        random_state=random_state,
        n_jobs=-1
        )

def get_xgboost(random_state=42) -> XGBClassifier:
    """
    Returns an XGBoost model with scale_pos_weight = 1.
    """
    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=random_state,
        eval_metric="mlogloss",
    )

def get_xgb_sample_weight(y_train) -> np.ndarray:
    """
    Per-sample weights for XGBoost model, so XGBoost gets the same balanced class as the other two models.
    We need to pass this into .fit(X,y, sample_weight=...).
    Returns the sample weights for XGBoost model.
    """
    return compute_sample_weight(class_weight="balanced", y=y_train)

MODEL_REGISTRY = {
    "logistic_regression": get_logistic_regression,
    "random_forest": get_random_forest,
    "xgboost": get_xgboost,
}