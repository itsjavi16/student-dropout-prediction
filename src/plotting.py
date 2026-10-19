OUTCOME_ORDER = ["Dropout", "Enrolled", "Graduate"]

OUTCOME_PALETTE = {
    "Dropout": "#eb6834",  # orange
    "Enrolled": "#1baf7a",  # aqua
    "Graduate": "#2a78d6",  # blue
}


def outcome_colors(labels):
    """List of outcome colors in the same order as `labels` (e.g. DataFrame columns)."""
    return [OUTCOME_PALETTE[label] for label in labels]


# Model colors: deliberately different hues from the outcome colors above, so a
# color always means the same thing across the deck (validated for color-blind
# readers; yellow and pink are light, so charts using them should label values).
MODEL_ORDER = ["Logistic Regression", "Random Forest", "XGBoost"]

MODEL_PALETTE = {
    "Logistic Regression": "#6250d6",  # violet
    "Random Forest": "#e87ba4",  # pink
    "XGBoost": "#eda100",  # gold
}
