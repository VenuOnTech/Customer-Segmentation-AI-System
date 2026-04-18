import shap
import pandas as pd

def generate_shap_explanations(model, X):

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    explanations = []

    for i in range(len(X)):
        feature_impact = dict(zip(X.columns, shap_values[1][i]))

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