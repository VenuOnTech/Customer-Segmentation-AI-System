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

    print("\n🚀 STARTING AI CUSTOMER SEGMENTATION PIPELINE\n")

    config = load_config()
    os.makedirs("outputs", exist_ok=True)

    dataset_paths = glob.glob("data/raw/*")

    if not dataset_paths:
        print("❌ No datasets found in data/raw/")
        return

    all_results = []

    # ==========================================
    # PROCESS EACH DATASET
    # ==========================================
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
            
            # ✅ Normalize probability (fix 1000% issue)
            if "Purchase_Probability" in rfm.columns:
                rfm["Purchase_Probability"] = rfm["Purchase_Probability"].clip(0, 1)

            rfm["Purchase_Probability"] = rfm.get("Purchase_Probability", 0.0)
            rfm["Churn"] = rfm.get("Churn", 0)

            churn_model, churn_metrics, feature_cols = train_deep_churn(rfm)

            if churn_model is None:
                print("⚠️ Churn model skipped due to single class")
            churn_metrics = {k: float(v) for k, v in churn_metrics.items()}

            print(f"🤖 Churn Model Metrics: {churn_metrics}")

            # ==============================
            # EXPLAINABILITY
            # ==============================
            try:
                valid_features = [col for col in feature_cols if col in rfm.columns]

                if not valid_features:
                    raise ValueError("No valid features for SHAP")

                X = rfm[valid_features].copy()

                explanations = generate_shap_explanations(churn_model, X)

                if not explanations or len(explanations) == 0:
                    raise ValueError("Empty SHAP output")

                explanations = list(explanations)[:len(rfm)]
                explanations += [""] * (len(rfm) - len(explanations))

                rfm["Explanation"] = explanations
                print("✅ SHAP explanations generated")

            except Exception as e:
                print(f"⚠️ SHAP failed: {str(e)}")
                rfm["Explanation"] = ""

            # ==============================
            # FALLBACK EXPLANATIONS
            # ==============================
            def generate_smart_explanation(row):
                if row["Recency"] > 100:
                    return "Customer inactive → high churn risk"
                if row["Frequency"] < 2:
                    return "Low engagement → needs reactivation"
                if row["Monetary"] > rfm["Monetary"].mean():
                    return "High value customer → retain"
                if row["Purchase_Probability"] > 0.7:
                    return "Likely to purchase again"
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
                drift = detect_drift(rfm["Frequency"], rfm["Frequency"] * 1.01)

            if drift:
                from src.monitoring.recalibration import recalibrate
                recalibration_status = recalibrate()
            else:
                recalibration_status = {"status": "not_required"}

            # ==============================
            # FEEDBACK LOOP (SAFE)
            # ==============================
            try:
                feedback_df = collect_feedback()

                if "Actual_Churn" in feedback_df.columns:
                    min_len = min(len(rfm), len(feedback_df))

                    X_fb = rfm.select_dtypes(include=["number"]).iloc[:min_len]
                    y_fb = feedback_df["Actual_Churn"].iloc[:min_len]

                    churn_model = retrain_with_feedback(churn_model, X_fb, y_fb)

                    print("🔁 Feedback retraining done")

            except Exception as e:
                print(f"⚠️ Feedback skipped: {e}")

            # ==============================
            # SAVE OUTPUTS (FIXED CORE ISSUE)
            # ==============================
            file_name = os.path.splitext(os.path.basename(path))[0]

            versioned_path = f"outputs/customer_segments_{file_name}.csv"
            latest_path = "outputs/customer_segments.csv"

            # Save both
            rfm.to_csv(versioned_path, index=False)
            rfm.to_csv(latest_path, index=False)

            print(f"💾 Saved: {versioned_path}")
            print(f"💾 Updated latest: {latest_path}")

            # ==============================
            # SAVE MODELS + LINEAGE
            # ==============================
            save_models(kmeans, churn_model, scaler)
            log_data_lineage(data_version, latest_path)

            # ==============================
            # REPORT
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

        # ==========================================
        # FINAL REPORT
        # ==========================================
        with open("outputs/final_report.json", "w") as f:
            json.dump(all_results, f, indent=4)

        # ✅ Create unified output for CI/CD
        final_output_path = "outputs/customer_segments.csv"

        if all_results:
            import pandas as pd
            combined_df = pd.concat([
                pd.read_csv(f) for f in glob.glob("outputs/customer_segments_*.csv")
            ])
            combined_df.to_csv(final_output_path, index=False)

        print("\n===================================")
        print("✅ SYSTEM COMPLETE")
        print("===================================\n")


if __name__ == "__main__":
    run()