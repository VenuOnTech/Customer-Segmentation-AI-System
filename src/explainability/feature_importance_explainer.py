import numpy as np

def generate_feature_importance(model, X):

    # Works for tree-based & linear models
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return ["No explanation available"] * len(X)

    explanations = []

    for i in range(len(X)):
        row_values = X.iloc[i]

        feature_impact = {
            col: float(val * imp)
            for col, val, imp in zip(X.columns, row_values, importances)
        }

        top_features = sorted(
            feature_impact.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:2]

        explanation = ", ".join(
            [f"{feat}: {round(val,2)}" for feat, val in top_features]
        )

        explanations.append(explanation)

    return explanations