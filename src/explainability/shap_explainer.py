import shap
import numpy as np

def generate_shap_explanations(model, X):

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    explanations = []

    # Handle different SHAP formats
    if isinstance(shap_values, list):
        shap_array = shap_values[1]
    else:
        shap_array = shap_values

    for i in range(len(X)):
        values = shap_array[i]

        # ✅ FIX: force 1D + scalar values
        values = np.array(values).flatten()

        feature_impact = {
            col: float(val) for col, val in zip(X.columns, values)
        }

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


def explain_customer(row):
    return f"Customer with Frequency={row.get('Frequency', 'NA')} and Monetary={row.get('Monetary', 'NA')}"
