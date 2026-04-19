def apply_feature_weights(X):
    """
    Apply business-driven weights to features
    """

    weights = {
        # 🔥 Core business drivers
        "Monetary": 2.0,
        "Frequency": 1.8,
        "Recency": 1.5,

        # 🔹 Behavioral
        "Avg_Interpurchase_Time": 1.2,
        "Avg_Quantity": 1.2,
        "Avg_Price": 1.2,

        # 🔹 Temporal
        "Active_Months": 1.3,
        "Purchase_Consistency": 1.3
    }

    for col in X.columns:
        if col in weights:
            X[col] = X[col] * weights[col]

    return X