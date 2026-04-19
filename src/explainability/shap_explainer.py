import numpy as np


def generate_shap_explanations(model, X):

    try:
        import shap  # ✅ import inside (prevents CI crash at import time)

        # 🔹 Safety: limit rows (extra protection)
        MAX_ROWS = 500
        if len(X) > MAX_ROWS:
            X = X.sample(MAX_ROWS, random_state=42)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        explanations = []

        # 🔹 Handle different SHAP formats
        if isinstance(shap_values, list):
            shap_array = shap_values[1]
        else:
            shap_array = shap_values

        for i in range(len(X)):
            values = shap_array[i]

            # 🔹 Ensure clean numeric vector
            values = np.array(values).flatten()

            feature_impact = {
                col: float(val) for col, val in zip(X.columns, values)
            }

            # 🔹 Top 2 important features
            top_features = sorted(
                feature_impact.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:2]

            explanation = ", ".join(
                [f"{feat} impact: {round(val, 2)}" for feat, val in top_features]
            )

            explanations.append(explanation)

        return explanations

    except Exception as e:
        print(f"⚠️ SHAP failed: {e}")
        return ["Explanation unavailable"] * len(X)


def explain_customer(row):
    return f"Customer with Frequency={row.get('Frequency', 'NA')} and Monetary={row.get('Monetary', 'NA')}"