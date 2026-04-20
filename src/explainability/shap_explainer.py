import shap
import numpy as np


def generate_shap_explanations(model, X, max_samples=500):

    # ==============================
    # 🔹 SAMPLING (VERY IMPORTANT)
    # ==============================
    if len(X) > max_samples:
        print(f"⚠️ SHAP sampling: {len(X)} → {max_samples}")
        X_sample = X.sample(max_samples, random_state=42)
    else:
        X_sample = X

    X_sample = X_sample.astype(float)

    try:
        # ==============================
        # 🔥 CASE 1: TREE MODELS
        # ==============================
        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

            if isinstance(shap_values, list):
                shap_array = shap_values[1]
            else:
                shap_array = shap_values

        # ==============================
        # 🔥 CASE 2: NEURAL NETWORK / MLP
        # ==============================
        else:
            print("⚠️ Using KernelExplainer (slow but works for MLP)")

            background = shap.sample(X_sample, 50)

            explainer = shap.KernelExplainer(
                model.predict_proba,
                background
            )

            shap_array = explainer.shap_values(X_sample, nsamples=100)

            if isinstance(shap_array, list):
                shap_array = shap_array[1]

        # ==============================
        # 🔹 GENERATE TEXT EXPLANATIONS
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

        # ==============================
        # 🔹 MAP BACK TO FULL DATA
        # ==============================
        full_explanations = ["Not computed"] * len(X)

        for idx, exp in zip(X_sample.index, explanations):
            full_explanations[idx] = exp

        return full_explanations

    except Exception as e:
        print(f"⚠️ SHAP failed → fallback used: {e}")

        return ["Fallback explanation"] * len(X)