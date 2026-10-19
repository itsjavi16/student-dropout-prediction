OUTCOME_ORDER = ["Dropout", "Enrolled", "Graduate"]

OUTCOME_PALETTE = {
    "Dropout": "#eb6834",  # orange
    "Enrolled": "#1baf7a",  # aqua
    "Graduate": "#2a78d6",  # blue
}


def outcome_colors(labels):
    """List of outcome colors in the same order as `labels` (e.g. DataFrame columns)."""
    return [OUTCOME_PALETTE[label] for label in labels]
