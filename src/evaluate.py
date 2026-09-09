"""
Shared evaluation utilities so all three models are compared consistently.
"""
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
 
 
def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    return {
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
    }
 
 
def evaluate_feature_set(
    df,
    feature_cols: list,
    feature_set_name: str,
    model,
    scale_numeric: bool,
    target_col: str = "Target",
    classes: list = ("Dropout", "Enrolled", "Graduate"),
    test_size: float = 0.2,
    random_state: int = 42,
    return_split: bool = False,
):
    """
    Fit `model` using one set of available predictors (via
    src.preprocessing.build_model_pipeline) and return overall + per-class
    metrics, which are used for the feature-availability comparison (Day 1 / After
    Semester 1 / Full Data), and shared between any model so Logistic
    Regression and Random Forest (and XGBoost) are compared identically.
 
    Same random_state + same n rows + same stratify=y means every model
    gets the SAME row split for a given feature set, so results are
    directly comparable across models.
 
    Set return_split=True to also get back (pipe, X_test, y_test, y_pred).
    Needed when a later step (e.g. a per-class recall comparison) has
    to reuse this exact fitted pipeline and test set rather than refitting.
    """
    from src.preprocessing import build_model_pipeline  # local import avoids a cycle
 
    X = df[feature_cols]
    y = df[target_col]
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
 
    pipe = build_model_pipeline(model=model, feature_cols=feature_cols, scale_numeric=scale_numeric)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
 
    per_class_recall = recall_score(y_test, y_pred, labels=list(classes), average=None)
 
    results = {
        "Feature Set": feature_set_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Macro F1": f1_score(y_test, y_pred, average="macro"),
        **{f"{cls} Recall": r for cls, r in zip(classes, per_class_recall)},
    }
 
    if return_split:
        return results, pipe, X_test, y_test, y_pred
    return results
 
 
def get_feature_importance(model, feature_names):
    """
    Returns a sorted (feature, importance) list. Works for models with
    .coef_ (logistic regression, take abs + average across classes) or
    .feature_importances_ (random forest, xgboost).
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = abs(model.coef_).mean(axis=0)
    else:
        raise ValueError("Model has neither feature_importances_ nor coef_")
 
    pairs = list(zip(feature_names, importances))
    return sorted(pairs, key=lambda x: x[1], reverse=True)