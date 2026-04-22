import shap
import numpy as np


def generate_shap_explanations(model, X, max_samples=500):

    # ==============================
    # SAMPLING
    # ==============================
    if len(X) > max_samples:
        print(f"⚠️ SHAP sampling: {len(X)} → {max_samples}")
        X_sample = X.sample(max_samples, random_state=42)
    else:
        X_sample = X

    X_sample = X_sample.astype(float)

    try:
        # ==============================
        # MODEL TYPE HANDLING
        # ==============================
        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            shap_array = shap_values[1] if isinstance(shap_values, list) else shap_values

        else:
            print("⚠️ Using KernelExplainer (MLP fallback)")

            background = shap.sample(X_sample, min(50, len(X_sample)))

            explainer = shap.KernelExplainer(
                model.predict_proba,
                background
            )

            shap_values = explainer.shap_values(X_sample, nsamples=50)
            shap_array = shap_values[1] if isinstance(shap_values, list) else shap_values

        # ==============================
        # INTERPRETATION FUNCTION
        # ==============================
        def interpret_feature(feat, val):
            direction = "high" if val > 0 else "low"

            if "Recency" in feat:
                return f"{direction} recency"
            elif "Frequency" in feat:
                return f"{direction} purchase frequency"
            elif "Monetary" in feat:
                return f"{direction} spending"
            elif "Lifetime" in feat:
                return f"{direction} customer lifetime"
            elif "Velocity" in feat:
                return f"{direction} purchase velocity"
            else:
                return f"{direction} {feat.lower()}"

        # ==============================
        # GENERATE EXPLANATIONS
        # ==============================
        explanations = []

        for i in range(len(X_sample)):

            values = np.array(shap_array[i]).flatten()

            feature_impact = {
                col: float(val)
                for col, val in zip(X_sample.columns, values)
            }

            # 🔥 FILTER meaningful features
            top_features = [
                (feat, val)
                for feat, val in sorted(
                    feature_impact.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )
                if abs(val) > 0.05
            ][:2]

            if len(top_features) == 0:
                explanation = "Not computed"
            else:
                interpreted = [interpret_feature(f, v) for f, v in top_features]

                if any(v > 0 for _, v in top_features):
                    explanation = "Churn risk influenced by " + " & ".join(interpreted)
                else:
                    explanation = "Stable customer due to " + " & ".join(interpreted)

            explanations.append(explanation)

        # ==============================
        # MAP BACK TO FULL DATA
        # ==============================
        full_explanations = [""] * len(X)

        for idx, exp in zip(X_sample.index, explanations):
            full_explanations[idx] = exp

        return full_explanations

    except Exception as e:
        print(f"⚠️ SHAP failed → fallback used: {e}")
        return ["Fallback explanation"] * len(X)


# ==========================================
# SINGLE CUSTOMER EXPLANATION
# ==========================================
def explain_customer(model, X_single):

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