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
from src.feature_engineering.temporal_features import add_temporal_features
from src.feature_engineering.behavioral_features import add_behavioral_features
from src.feature_engineering.autoencoder_features import generate_autoencoder_features

from src.segmentation.kmeans_segmentation import run_kmeans

from src.prediction.deep_churn_model import train_deep_churn
from src.prediction.future_prediction import predict_future_purchase
from src.prediction.lstm_churn_model import train_lstm_churn, predict_lstm

from src.streaming.stream_simulator import stream_data

from src.monitoring.behavior_drift import detect_drift
from src.monitoring.data_validation import validate_data
from src.monitoring.data_lineage import log_data_lineage

from src.model_management.model_versioning import save_models
from src.utils.config_loader import load_config
from src.utils.experiment_tracker import log_experiment
from src.feature_store.store import save_features

from src.explainability.shap_explainer import generate_shap_explanations
from src.data_ingestion.data_versioning import get_data_version


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

            # ==========================================
            # ⚡ SAMPLING (ONLY FOR LITE MODE)
            # ==========================================
            if config.get("mode") == "lite" and len(df) > 50000:
                df = df.sample(n=50000, random_state=42)
                print("⚡ Lite mode: using sampled dataset")

            # ==========================================
            # STREAM PROCESSING (REAL-TIME SIMULATION)
            # ==========================================
            all_batches = []

            for batch in stream_data(df):

                mapping = detect_columns(batch)
                validate_data(batch, mapping, strict=False)

                batch = add_multi_source_features(batch)
                batch = clean_data(batch, mapping)

                validate_data(batch, mapping, strict=True)

                temporal = add_temporal_features(batch, mapping)
                rfm_batch = create_rfm(batch, mapping)
                behavioral = add_behavioral_features(batch, mapping)

                rfm_batch = rfm_batch.merge(behavioral, on="CustomerID", how="left")
                rfm_batch = rfm_batch.merge(temporal, on="CustomerID", how="left")
                rfm_batch = rfm_batch.fillna(0)

                all_batches.append(rfm_batch)

            # Combine streaming results
            rfm = pd.concat(all_batches, ignore_index=True)
            # ✅ SAFE AGGREGATION AFTER STREAMING
            numeric_cols = rfm.select_dtypes(include=["number"]).columns.tolist()

            rfm = rfm.groupby("CustomerID", as_index=False)[numeric_cols].sum()

            print(f"🧠 Combined RFM shape: {rfm.shape}")

            # ==========================================
            # DATA VERSIONING
            # ==========================================
            data_version = get_data_version(df)

            # ==========================================
            # AUTOENCODER FEATURES
            # ==========================================
            try:
                rfm = generate_autoencoder_features(rfm)
                print("✅ Autoencoder features added")
            except Exception as e:
                print(f"⚠️ Autoencoder skipped: {e}")

            # ==========================================
            # LSTM (ONLY FULL MODE)
            # ==========================================
            if config.get("mode") == "full":
                try:
                    lstm_model, scaler = train_lstm_churn(rfm)
                    rfm = predict_lstm(lstm_model, scaler, rfm)
                    print("✅ LSTM enabled")
                except Exception as e:
                    print(f"⚠️ LSTM skipped: {e}")
                    rfm["LSTM_Score"] = 0
            else:
                rfm["LSTM_Score"] = 0

            # ==========================================
            # FEATURE STORE
            # ==========================================
            save_features(rfm)

            # ==========================================
            # SEGMENTATION
            # ==========================================
            rfm, kmeans, scaler_kmeans, metrics = run_kmeans(rfm, config)

            if "Final_Cluster" in rfm.columns:
                rfm["Cluster"] = rfm["Final_Cluster"]

            print(f"📊 Clustering Metrics: {metrics}")

            # ==========================================
            # PURCHASE PREDICTION
            # ==========================================
            rfm = predict_future_purchase(rfm)
            rfm["Purchase_Probability"] = rfm["Purchase_Probability"].clip(0, 1)

            # ==========================================
            # CHURN MODEL
            # ==========================================
            churn_model, churn_metrics, feature_cols = train_deep_churn(rfm)

            churn_metrics = {k: float(v) for k, v in churn_metrics.items()}
            print(f"🤖 Churn Model Metrics: {churn_metrics}")

            # ==========================================
            # EXPLAINABILITY (ONLY FULL MODE)
            # ==========================================
            if config.get("mode") == "full" and churn_model is not None:
                try:
                    X = rfm[feature_cols]
                    rfm["Explanation"] = generate_shap_explanations(churn_model, X)
                    print("✅ SHAP enabled")
                except Exception as e:
                    print(f"⚠️ SHAP failed: {e}")
                    rfm["Explanation"] = ""
            else:
                rfm["Explanation"] = ""

            # ==========================================
            # FALLBACK EXPLANATIONS
            # ==========================================
            def explain(row):
                if row["Churn"] == 1:
                    return "Inactive → churn risk"
                if row["LSTM_Score"] > 0.7:
                    return "High engagement customer"
                if row["Monetary"] > rfm["Monetary"].mean():
                    return "High value customer"
                if row["Frequency"] < rfm["Frequency"].quantile(0.25):
                    return "Low engagement → needs attention"
                return "Moderate customer"

            rfm["Explanation"] = rfm.apply(
                lambda row: explain(row) if row["Explanation"] == "" else row["Explanation"],
                axis=1
            )

            # ==========================================
            # DRIFT DETECTION
            # ==========================================
            drift = detect_drift(rfm["Frequency"], "Frequency")

            # ==========================================
            # SAVE OUTPUTS
            # ==========================================
            file_name = os.path.splitext(os.path.basename(path))[0]

            versioned_path = f"outputs/customer_segments_{file_name}.csv"
            latest_path = "outputs/customer_segments.csv"

            rfm.to_csv(versioned_path, index=False)
            rfm.to_csv(latest_path, index=False)

            save_models(kmeans, churn_model, scaler_kmeans)
            log_data_lineage(data_version, latest_path)

            # ==========================================
            # LOG EXPERIMENT
            # ==========================================
            log_experiment(
                params=config,
                metrics={"clustering": metrics, "churn": churn_metrics}
            )

            all_results.append({
                "dataset": file_name,
                "rows": int(len(rfm)),
                "clustering": metrics,
                "churn": churn_metrics,
                "drift_detected": bool(drift)
            })

        except Exception as e:
            print(f"❌ Error processing {path}: {str(e)}")

    # ==========================================
    # FINAL REPORT
    # ==========================================
    with open("outputs/final_report.json", "w") as f:
        json.dump(all_results, f, indent=4)

    print("\n===================================")
    print("✅ SYSTEM COMPLETE")
    print("===================================\n")


if __name__ == "__main__":
    run()