import numpy as np


def generate_feature_importance_explanations(model, X):

    explanations = []

    try:
        # 🔹 Tree-based models (preferred)
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_

        # 🔹 Linear / MLP fallback
        elif hasattr(model, "coef_"):
            importances = np.mean(np.abs(model.coef_), axis=0)

        else:
            # 🔹 Ultimate fallback
            importances = np.ones(X.shape[1])

        feature_names = list(X.columns)

        for i in range(len(X)):

            values = X.iloc[i].values

            feature_impact = {
                col: float(val * imp)
                for col, val, imp in zip(feature_names, values, importances)
            }

            top_features = sorted(
                feature_impact.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:2]

            explanation = ", ".join(
                [f"{feat}: {round(val, 2)}" for feat, val in top_features]
            )

            explanations.append(explanation)

    except Exception as e:
        print(f"⚠️ Explainability fallback used: {e}")
        explanations = ["No explanation available"] * len(X)

    return explanations