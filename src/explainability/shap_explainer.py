import shap
import numpy as np

def generate_shap_explanations(model, X, max_samples=2000):

    # 🔹 SAMPLE DATA (CRITICAL FIX)
    if len(X) > max_samples:
        print(f"⚠️ SHAP sampling: {len(X)} → {max_samples}")
        X_sample = X.sample(max_samples, random_state=42)
        sample_indices = X_sample.index
    else:
        X_sample = X
        sample_indices = X.index

    # 🔹 Use safe explainer
    try:
        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample)
        shap_array = shap_values.values
    except Exception:
        # fallback (more stable for tree models)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        if isinstance(shap_values, list):
            shap_array = shap_values[1]
        else:
            shap_array = shap_values

    explanations_map = {}

    for i, idx in enumerate(sample_indices):

        values = np.array(shap_array[i]).flatten()

        feature_impact = {
            col: float(val) for col, val in zip(X_sample.columns, values)
        }

        top_features = sorted(
            feature_impact.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:2]

        explanation = ", ".join(
            [f"{feat} impact: {round(val, 2)}" for feat, val in top_features]
        )

        explanations_map[idx] = explanation

    # 🔹 Fill remaining rows with default explanation
    final_explanations = []
    for idx in X.index:
        if idx in explanations_map:
            final_explanations.append(explanations_map[idx])
        else:
            final_explanations.append("Default explanation (sampled)")

    return final_explanations


def explain_customer(row):
    return f"Customer with Frequency={row.get('Frequency', 'NA')} and Monetary={row.get('Monetary', 'NA')}"