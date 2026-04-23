import numpy as np
import json
import os
import glob
import pandas as pd

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
from src.feature_store.store import save_features
from src.utils.experiment_tracker import log_experiment


def run():
    print("\n🚀 STARTING AI CUSTOMER SEGMENTATION PIPELINE\n")

    config = load_config()
    os.makedirs("outputs", exist_ok=True)

    dataset_paths = glob.glob("data/raw/*")

    if not dataset_paths:
        print("❌ No datasets found in data/raw/")
        return

    all_results = []

    for path in dataset_paths:

        print(f"\n📂 Processing dataset: {path}")

        try:
            df = load_data(path)

            if df is None or df.empty:
                print("⚠️ Empty dataset, skipping")
                continue

            # ==============================
            # SCHEMA + VALIDATION
            # ==============================
            mapping = detect_columns(df)
            validate_data(df, mapping, strict=False)

            df = add_multi_source_features(df)
            df = clean_data(df, mapping)

            validate_data(df, mapping, strict=True)

            data_version = get_data_version(df)

            print(f"📊 Cleaned data shape: {df.shape}")

            # ==============================
            # FEATURE ENGINEERING
            # ==============================
            temporal = add_temporal_features(df, mapping)
            rfm = create_rfm(df, mapping)
            behavioral = add_behavioral_features(df, mapping)

            rfm = rfm.merge(behavioral, on="CustomerID", how="left")
            rfm = rfm.merge(temporal, on="CustomerID", how="left")
            rfm = rfm.fillna(0)

            print(f"🧠 RFM shape: {rfm.shape}")

            # ==============================
            # FEATURE STORE
            # ==============================
            save_features(rfm)

            if os.path.exists("outputs/feature_store/rfm_features_v1.csv"):
                print("✅ Feature store saved successfully")
            else:
                print("❌ Feature store NOT created")

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

            if "Purchase_Probability" in rfm.columns:
                rfm["Purchase_Probability"] = rfm["Purchase_Probability"].clip(0, 1)
            else:
                rfm["Purchase_Probability"] = 0.0

            # ==============================
            # CHURN TRAINING (SOURCE OF TRUTH)
            # ==============================
            churn_model, churn_metrics, feature_cols = train_deep_churn(rfm)

            churn_metrics = {k: float(v) for k, v in churn_metrics.items()}
            print(f"🤖 Churn Model Metrics: {churn_metrics}")

            # ==============================
            # EXPLAINABILITY
            # ==============================
            if config.get("mode") == "full" and churn_model is not None:
                try:
                    valid_features = [col for col in feature_cols if col in rfm.columns]

                    if not valid_features:
                        raise ValueError("No valid features")

                    X = rfm[valid_features].copy()
                    explanations = generate_shap_explanations(churn_model, X)

                    explanations = list(explanations)[:len(rfm)]
                    explanations += [""] * (len(rfm) - len(explanations))

                    rfm["Explanation"] = explanations
                    print("✅ SHAP explanations generated")

                except Exception as e:
                    print(f"⚠️ SHAP failed: {str(e)}")
                    rfm["Explanation"] = ""
            else:
                print("⚡ Lite mode → skipping SHAP")
                rfm["Explanation"] = ""

            # ==============================
            # SMART FALLBACK EXPLANATIONS
            # ==============================
            recency_threshold = rfm["Recency"].quantile(0.75)
            freq_threshold = rfm["Frequency"].quantile(0.25)

            def generate_smart_explanation(row):

                if row["Churn"] == 1 and row["Recency"] > recency_threshold and row["Frequency"] < freq_threshold:
                    return "Inactive customer → high churn risk"

                if row["Monetary"] > rfm["Monetary"].mean() and row["Frequency"] > rfm["Frequency"].median():
                    return "High value loyal customer → prioritize retention"

                if row["Purchase_Probability"] > 0.6:
                    return "High likelihood of repeat purchase"

                if row["Churn"] == 0 and row["Frequency"] > rfm["Frequency"].median():
                    return "Active and stable customer"

                if row["Frequency"] < freq_threshold:
                    return "Low engagement → needs reactivation"

                return "Moderate activity customer"

            rfm["Explanation"] = rfm["Explanation"].fillna("").astype(str)

            rfm["Explanation"] = rfm.apply(
                lambda row: generate_smart_explanation(row)
                if row["Explanation"].strip() == ""
                else row["Explanation"],
                axis=1
            )

            # ==============================
            # DRIFT DETECTION
            # ==============================
            drift = False
            if "Frequency" in rfm.columns:
                drift = detect_drift(rfm["Frequency"], feature_name="Frequency")

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

                if feedback_df is not None:
                    merged = rfm.merge(feedback_df, on="CustomerID", how="inner")

                    if len(merged) > 10:
                        X_fb = merged.select_dtypes(include=["number"]).drop(columns=["Actual_Churn"])
                        y_fb = merged["Actual_Churn"]

                        churn_model = retrain_with_feedback(churn_model, X_fb, y_fb)
                        print("🔁 Feedback retraining done")

            except Exception as e:
                print(f"⚠️ Feedback skipped: {e}")

            # ==============================
            # SAVE OUTPUTS
            # ==============================
            file_name = os.path.splitext(os.path.basename(path))[0]

            versioned_path = f"outputs/customer_segments_{file_name}.csv"
            latest_path = "outputs/customer_segments.csv"

            rfm.to_csv(versioned_path, index=False)
            rfm.to_csv(latest_path, index=False)

            print(f"💾 Saved: {versioned_path}")

            # ==============================
            # SAVE MODELS + LINEAGE
            # ==============================
            save_models(kmeans, churn_model, scaler)
            log_data_lineage(data_version, latest_path)

            # ==============================
            # LOG EXPERIMENT
            # ==============================
            log_experiment(
                params=config,
                metrics={
                    "clustering": metrics,
                    "churn": churn_metrics
                }
            )

            # ==============================
            # REPORT ENTRY
            # ==============================
            all_results.append({
                "dataset": file_name,
                "rows": int(len(rfm)),
                "clustering": metrics,
                "churn": churn_metrics,
                "drift_detected": bool(drift),
                "recalibration": recalibration_status
            })

        except Exception as e:
            print(f"❌ Error processing {path}: {str(e)}")

    # ==============================
    # SAVE REPORTS
    # ==============================
    with open("outputs/experiments.json", "w") as f:
        json.dump(all_results, f, indent=4)

    with open("outputs/final_report.json", "w") as f:
        json.dump(all_results, f, indent=4)

    if all_results:
        combined_df = pd.concat([
            pd.read_csv(f) for f in glob.glob("outputs/customer_segments_*.csv")
        ])
        combined_df.to_csv("outputs/customer_segments.csv", index=False)

    print("\n===================================")
    print("✅ SYSTEM COMPLETE")
    print("===================================\n")


if __name__ == "__main__":
    run()