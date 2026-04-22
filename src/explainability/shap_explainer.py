import shap
import numpy as np


def generate_shap_explanations(model, X, max_samples=500):

    if len(X) > max_samples:
        print(f"⚠️ SHAP sampling: {len(X)} → {max_samples}")
        X_sample = X.sample(max_samples, random_state=42)
    else:
        X_sample = X

    X_sample = X_sample.astype(float)

    try:
        # ==============================
        # TREE MODELS
        # ==============================
        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

            shap_array = shap_values[1] if isinstance(shap_values, list) else shap_values

        # ==============================
        # MLP / GENERIC MODELS
        # ==============================
        else:
            print("⚠️ Using KernelExplainer (MLP fallback)")

            background = shap.sample(X_sample, min(50, len(X_sample)))

            explainer = shap.KernelExplainer(
                model.predict_proba,
                background
            )

            shap_array = explainer.shap_values(X_sample, nsamples=50)

            shap_array = shap_array[1] if isinstance(shap_array, list) else shap_array

        # ==============================
        # TEXT EXPLANATIONS
        # ==============================
        explanations = []

        for i in range(len(X_sample)):

            values = np.array(shap_array[i]).flatten()

            feature_impact = {
                col: float(val)
                for col, val in zip(X_sample.columns, values)
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

        full_explanations = [""] * len(X)

        for idx, exp in zip(X_sample.index, explanations):
            full_explanations[idx] = exp

        return full_explanations

    except Exception as e:
        print(f"⚠️ SHAP failed → fallback used: {e}")
        return ["Fallback explanation"] * len(X)


# ==========================================
# ✅ REQUIRED FOR TESTS (CRITICAL FIX)
# ==========================================
def explain_customer(model, X_single):
    """
    Explain a single customer (used in tests)
    """
    try:
        X_single = X_single.astype(float)

        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_single)

            values = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

        else:
            background = shap.sample(X_single, min(10, len(X_single)))

            explainer = shap.KernelExplainer(
                model.predict_proba,
                background
            )

            shap_values = explainer.shap_values(X_single, nsamples=20)

            values = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

        feature_impact = {
            col: float(val)
            for col, val in zip(X_single.columns, values)
        }

        top_features = sorted(
            feature_impact.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:2]

        return ", ".join([f"{k}: {round(v, 2)}" for k, v in top_features])

    except Exception:
        return "Fallback explanation"