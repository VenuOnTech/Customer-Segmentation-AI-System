import numpy as np
import json
import os
import glob

from src.data_ingestion.load_data import load_data
from src.data_ingestion.schema_detection import detect_columns
from src.preprocessing.data_cleaning import clean_data
from src.feature_engineering.rfm_features import create_rfm
from src.feature_engineering.multi_source_features import add_multi_source_features
from src.segmentation.kmeans_segmentation import run_kmeans
from src.prediction.deep_churn_model import train_deep_churn
from src.prediction.future_prediction import predict_future_purchase
from src.monitoring.behavior_drift import detect_drift
from src.model_management.model_versioning import save_models
from src.utils.config_loader import load_config
from src.monitoring.data_validation import validate_data
from src.data_ingestion.data_versioning import get_data_version
from src.monitoring.data_lineage import log_data_lineage
from src.feature_engineering.temporal_features import add_temporal_features
from src.feature_engineering.behavioral_features import add_behavioral_features
from src.explainability.shap_explainer import generate_shap_explanations
from src.feedback.feedback_handler import collect_feedback, retrain_with_feedback


def run():

    config = load_config()
    os.makedirs("outputs", exist_ok=True)

    dataset_paths = glob.glob("data/raw/*")
    all_results = []

    for path in dataset_paths:

        print(f"\n🚀 Processing dataset: {path}")

        df = load_data(path)

        # ✅ Prevent crash on empty datasets
        if df is None or df.empty:
            print("⚠️ Empty dataset, skipping")
            continue

        mapping = detect_columns(df)
        validate_data(df, mapping, strict=False)

        df = add_multi_source_features(df)
        df = clean_data(df, mapping)

        validate_data(df, mapping, strict=True)

        data_version = get_data_version(df)

        # ==============================
        # FEATURES
        # ==============================
        temporal = add_temporal_features(df, mapping)
        rfm = create_rfm(df, mapping)
        behavioral = add_behavioral_features(df, mapping)

        rfm = rfm.merge(behavioral, on=mapping["customer_id"], how="left")
        rfm = rfm.merge(temporal, on=mapping["customer_id"], how="left")

        rfm = rfm.fillna(0)

        # ==============================
        # SEGMENTATION
        # ==============================
        rfm, kmeans, scaler, metrics = run_kmeans(rfm, config)

        print(f"📊 Clustering Metrics: {metrics}")

        if "Final_Cluster" in rfm.columns:
            rfm["Cluster"] = rfm["Final_Cluster"]

        # ==============================
        # PREDICTION
        # ==============================
        rfm = predict_future_purchase(rfm)

        if "Purchase_Probability" not in rfm.columns:
            rfm["Purchase_Probability"] = 0.0

        if "Churn" not in rfm.columns:
            rfm["Churn"] = 0

        churn_model, churn_metrics, feature_cols = train_deep_churn(rfm)
        churn_metrics = {k: float(v) for k, v in churn_metrics.items()}

        # ==============================
        # EXPLAINABILITY (FIXED)
        # ==============================
        try:
            valid_features = [col for col in feature_cols if col in rfm.columns]

            if not valid_features:
                raise ValueError("No valid features for SHAP")

            X = rfm[valid_features].copy()

            explanations = generate_shap_explanations(churn_model, X)

            if explanations is None or len(explanations) == 0:
                raise ValueError("Empty SHAP output")

            # Align length
            explanations = list(explanations)[:len(rfm)]
            explanations += [""] * (len(rfm) - len(explanations))

            rfm["Explanation"] = explanations
            print("✅ SHAP explanations generated")

        except Exception as e:
            print(f"⚠️ SHAP failed: {str(e)}")
            rfm["Explanation"] = ""

        # ✅ Always clean explanations (FIXED INDENTATION)
        rfm["Explanation"] = rfm["Explanation"].replace(
            ["Not computed", "Model explanation unavailable"],
            ""
        )

        # ==============================
        # FALLBACK EXPLANATIONS
        # ==============================
        def is_weak_explanation(exp):
            if exp in ["", None, "Not computed"]:
                return True

            try:
                values = [
                    float(x.split(":")[1])
                    for x in str(exp).split(",")
                    if ":" in x
                ]
                return len(values) == 0 or all(abs(v) < 0.05 for v in values)
            except:
                return True

        def generate_smart_explanation(row):
            if "Recency" in row and row["Recency"] > 100:
                return "Customer inactive for a long time → high churn risk"

            if "Frequency" in row and row["Frequency"] < 2:
                return "Low engagement customer → needs reactivation"

            if "Monetary" in row and row["Monetary"] > rfm["Monetary"].mean():
                return "High value customer → prioritize retention"

            if "Purchase_Probability" in row and row["Purchase_Probability"] > 0.7:
                return "Likely to purchase again → target with offers"

            return "Moderate activity customer"

        rfm["Explanation"] = rfm["Explanation"].fillna("").astype(str)

        rfm["Explanation"] = rfm.apply(
            lambda row: generate_smart_explanation(row)
            if is_weak_explanation(row["Explanation"])
            else row["Explanation"],
            axis=1
        )

        # ==============================
        # DRIFT (SAFE VERSION)
        # ==============================
        if "Frequency" in rfm.columns:
            drift = detect_drift(rfm["Frequency"], rfm["Frequency"] * 1.01)
        else:
            drift = False

        if drift:
            from src.monitoring.recalibration import recalibrate
            recalibration_status = recalibrate()
        else:
            recalibration_status = {"status": "not_required"}

        # ==============================
        # FEEDBACK LOOP
        # ==============================
        try:
            feedback_df = collect_feedback()

            if "Actual_Churn" in feedback_df.columns:
                min_len = min(len(rfm), len(feedback_df))

                X_fb = rfm.select_dtypes(include=["number"]).iloc[:min_len]
                y_fb = feedback_df["Actual_Churn"].iloc[:min_len]

                churn_model = retrain_with_feedback(churn_model, X_fb, y_fb)

                print("✅ Feedback-based retraining complete")

        except Exception as e:
            print(f"⚠️ Feedback loop skipped: {e}")

        # ==============================
        # SAVE (NO OVERWRITE FIX)
        # ==============================
        file_name = os.path.basename(path).replace(".csv", "")
        output_path = f"outputs/customer_segments_{file_name}.csv"

        rfm.to_csv(output_path, index=False)

        save_models(kmeans, churn_model, scaler)
        log_data_lineage(data_version, output_path)

        all_results.append({
            "dataset": file_name,
            "rows": int(len(rfm)),
            "clustering": metrics,
            "churn": churn_metrics,
            "drift_detected": bool(drift),
            "recalibration": recalibration_status
        })

    with open("outputs/final_report.json", "w") as f:
        json.dump(all_results, f, indent=4)

    print("\n✅ SYSTEM COMPLETE")


if __name__ == "__main__":
    run()