def apply_feature_weights(X, config=None):

    X = X.copy()  # 🔥 safety

    if config and "feature_weights" in config:
        weights = config["feature_weights"]
    else:
        weights = {}

    for col in X.columns:
        X[col] = X[col] * weights.get(col, 1.0)

    return X