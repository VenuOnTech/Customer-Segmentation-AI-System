import pandas as pd

def select_features(rfm):
    """
    Select only meaningful features for clustering
    """

    selected_features = [
        # Core RFM
        "Recency",
        "Frequency",
        "Monetary",

        # Behavioral
        "Avg_Interpurchase_Time",
        "Avg_Quantity",
        "Avg_Price",

        # Temporal (only strong ones)
        "Active_Months",
        "Purchase_Consistency"
    ]

    # Keep only available columns (safe check)
    selected_features = [col for col in selected_features if col in rfm.columns]

    return rfm[selected_features].copy()