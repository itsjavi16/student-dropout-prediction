"""
Shared evaluation utilities so all three models are compared consistently.
"""
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from src.models import get_xgboost, get_xgb_sample_weight


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    return {
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True
        ),
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
    metrics -- used for the feature-availability comparison (Day 1 / After
    Semester 1 / Full Data), and shared between any model so Logistic
    Regression and Random Forest (and XGBoost) are compared identically.

    Same random_state + same n rows + same stratify=y means every model
    gets the SAME row split for a given feature set, so results are
    directly comparable across models.

    Set return_split=True to also get back (pipe, X_test, y_test, y_pred)
    -- needed when a later step (e.g. a per-class recall comparison) has
    to reuse this exact fitted pipeline and test set rather than refitting.
    """
    from src.preprocessing import build_model_pipeline  # local import avoids a cycle

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    pipe = build_model_pipeline(
        model=model, feature_cols=feature_cols, scale_numeric=scale_numeric
    )
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


def evaluate_regression_feature_set(
    df,
    feature_cols: list,
    feature_set_name: str,
    model,
    scale_numeric: bool,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    return_split: bool = False,
):
    """
    Regression counterpart to evaluate_feature_set: fits `model` to
    predict a continuous target (e.g. a semester grade) using one set of
    available predictors, via src.preprocessing.build_model_pipeline.
    Used for the GPA-trajectory subquestions.

    No stratify (continuous target), otherwise the same shape: same
    random_state + same n rows -> same row split across different feature
    sets/models for a fair comparison.

    Set return_split=True to also get back (pipe, X_test, y_test, y_pred).
    """
    from src.preprocessing import build_model_pipeline  # local import avoids a cycle
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    pipe = build_model_pipeline(
        model=model, feature_cols=feature_cols, scale_numeric=scale_numeric
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    results = {
        "Feature Set": feature_set_name,
        "R2": r2_score(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred) ** 0.5,
        "MAE": mean_absolute_error(y_test, y_pred),
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

def evaluate_xgboost_feature_set(
    df,
    feature_cols,
    feature_set_name,
    target_col="Target",
    classes=("Dropout", "Enrolled", "Graduate"),
    test_size=0.2,
    random_state=42,
    return_split=False
):
    """
    Evaluate XGBoost on one feature-availability set using the same
    stratified 80/20 split and metrics used for the other classifiers.

    XGBoost receives class-balanced sample weights during training.
    """

    from src.preprocessing import build_model_pipeline
    # XGBoost expects numeric class labels.
    label_encoder = LabelEncoder()
    label_encoder.fit(classes)

    X = df[feature_cols]

    y_original = df[target_col]
    y = pd.Series(
        label_encoder.transform(y_original),
        index=y_original.index
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    # Tree-based models do not need numeric standardization.
    pipe = build_model_pipeline(
        model=get_xgboost(random_state=random_state),
        feature_cols=feature_cols,
        scale_numeric=False
    )

    # Balance classes using weights calculated from the training set only.
    sample_weights = get_xgb_sample_weight(y_train)

    # XGBoost is the "model" step inside the sklearn Pipeline.
    pipe.fit(
        X_train,
        y_train,
        model__sample_weight=sample_weights
    )

    y_pred = pipe.predict(X_test)

    # Convert predictions back to original class labels for reporting.
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(
        y_pred.astype(int)
    )

    per_class_recall = recall_score(
        y_test_labels,
        y_pred_labels,
        labels=list(classes),
        average=None
    )

    results = {
        "Feature Set": feature_set_name,
        "Accuracy": accuracy_score(
            y_test_labels,
            y_pred_labels
        ),
        "Macro F1": f1_score(
            y_test_labels,
            y_pred_labels,
            average="macro"
        ),
        "Macro Recall": recall_score(
            y_test_labels,
            y_pred_labels,
            average="macro"
        ),
        **{
            f"{cls} Recall": r
            for cls, r in zip(classes, per_class_recall)
        }
    }

    if return_split:
        return (
            results,
            pipe,
            X_test,
            y_test_labels,
            y_pred_labels
        )

    return results

def cross_validate_xgboost(
    df,
    feature_cols,
    feature_set_name,
    target_col="Target",
    classes=("Dropout", "Enrolled", "Graduate"),
    n_splits=5,
    random_state=42
):
    from src.preprocessing import build_model_pipeline

    label_encoder = LabelEncoder()
    label_encoder.fit(classes)

    X = df[feature_cols]

    y_original = df[target_col]
    y = pd.Series(
        label_encoder.transform(y_original),
        index=y_original.index
    )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X, y),
        start=1
    ):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        pipe = build_model_pipeline(
            model=get_xgboost(random_state=random_state),
            feature_cols=feature_cols,
            scale_numeric=False
        )

        # Recalculate class-balanced weights using only the training portion
        # of each fold to avoid using validation-fold information.
        sample_weights = get_xgb_sample_weight(y_train)

        pipe.fit(
            X_train,
            y_train,
            model__sample_weight=sample_weights
        )

        y_pred = pipe.predict(X_val)

        y_val_labels = label_encoder.inverse_transform(y_val)
        y_pred_labels = label_encoder.inverse_transform(
            y_pred.astype(int)
        )

        class_recalls = recall_score(
            y_val_labels,
            y_pred_labels,
            labels=list(classes),
            average=None
        )

        fold_results.append({
            "Feature Set": feature_set_name,
            "Fold": fold,
            "Accuracy": accuracy_score(
                y_val_labels,
                y_pred_labels
            ),
            "Macro F1": f1_score(
                y_val_labels,
                y_pred_labels,
                average="macro"
            ),
            "Macro Recall": recall_score(
                y_val_labels,
                y_pred_labels,
                average="macro"
            ),
            **{
                f"{cls} Recall": r
                for cls, r in zip(classes, class_recalls)
            }
        })

    return pd.DataFrame(fold_results)