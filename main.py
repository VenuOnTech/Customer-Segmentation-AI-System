from src.data_ingestion.load_data import load_data
from src.data_ingestion.schema_detection import detect_columns
from src.preprocessing.data_cleaning import clean_data
from src.feature_engineering.rfm_features import create_rfm
from src.feature_engineering.multi_source_features import add_multi_source_features
from src.segmentation.kmeans_segmentation import run_kmeans
from src.prediction.churn_prediction import train_churn
from src.prediction.future_prediction import predict_future_purchase
from src.explainability.shap_explainer import generate_shap_explanations
from src.monitoring.behavior_drift import detect_drift
from src.model_management.model_versioning import save_models
from src.utils.config_loader import load_config
from src.feedback.feedback_handler import collect_feedback, retrain_with_feedback
from src.monitoring.data_quality import generate_data_quality_report
from src.monitoring.data_validation import validate_data
from src.data_ingestion.data_versioning import get_data_version
from src.monitoring.data_lineage import log_data_lineage
from src.feature_engineering.temporal_features import add_temporal_features
from src.feature_engineering.behavioral_features import add_behavioral_features

import json
import numpy as np
import os


def run():

    config = load_config()

    # 🔹 Ensure output directory exists FIRST
    os.makedirs("outputs", exist_ok=True)

    # 🔹 Load Data
    df = load_data("data/raw/Online_Retail.xlsx")

    # 🔹 Save snapshot
    df.sample(min(1000, len(df))).to_csv("outputs/data_snapshot.csv", index=False)

    # 🔹 Detect Schema
    mapping = detect_columns(df)

    # 🔹 Soft Validation (before cleaning)
    validate_data(df, mapping, strict=False)

    # 🔹 Add Multi-source Features
    df = add_multi_source_features(df)

    # 🔹 Clean Data
    df = clean_data(df, mapping)

    # 🔹 Data Versioning
    data_version = get_data_version(df)
    print(f"Data Version: {data_version}")

    # 🔹 Strict Validation (after cleaning)
    validate_data(df, mapping, strict=True)

    # ==============================
    # 🔥 FEATURE ENGINEERING BLOCK
    # ==============================

    # 🔹 Temporal Features (customer-level)
    temporal_features = add_temporal_features(df, mapping)

    # 🔹 RFM Features
    rfm = create_rfm(df, mapping)

    # 🔹 Behavioral Features
    behavioral = add_behavioral_features(df, mapping)

    # 🔹 Merge all features
    rfm = rfm.merge(behavioral, on=mapping["customer_id"], how="left")
    rfm = rfm.merge(temporal_features, on=mapping["customer_id"], how="left")

    # 🔹 Final NA handling before ML
    rfm = rfm.fillna(0)

    # ==============================
    # 🔹 SEGMENTATION
    # ==============================

    rfm, kmeans, scaler = run_kmeans(rfm, config)

    # 🔹 Future Prediction
    rfm = predict_future_purchase(rfm)

    # 🔹 Churn Model
    churn_model, churn_metrics = train_churn(rfm)

    # 🔹 Explainability
    X_explain = rfm[["Frequency", "Monetary"]]
    rfm["Explanation"] = generate_shap_explanations(churn_model, X_explain)

    # 🔹 Drift Detection
    old_data = rfm["Frequency"]
    noise = np.random.normal(0, 0.01, len(rfm))
    new_data = rfm["Frequency"] * (1 + noise)

    if detect_drift(old_data, new_data):
        print("Drift detected → retraining needed")

    # 🔥 SAVE MODELS
    save_models(kmeans, churn_model, scaler)

    # 🔥 SAVE OUTPUTS
    output_path = "outputs/customer_segments.csv"
    rfm.to_csv(output_path, index=True)

    print("Results saved to outputs/customer_segments.csv")

    # 🔹 Data Lineage
    log_data_lineage(data_version, output_path)

    # 🔁 FEEDBACK LOOP
    feedback_df = collect_feedback(output_path)

    X_feedback = feedback_df[["Recency", "Frequency", "Monetary"]]
    y_feedback = feedback_df["Actual_Churn"]

    churn_model = retrain_with_feedback(churn_model, X_feedback, y_feedback)

    # 🔹 Save Model Performance
    with open("outputs/model_performance.json", "w") as f:
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            return str(obj)

        json.dump(churn_metrics, f, indent=4, default=convert_to_serializable)

    print("Model performance saved")
    
    # 🔹 Data Quality Report
    quality_report = generate_data_quality_report(df)

    with open("outputs/data_quality_report.json", "w") as f:

        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            return str(obj)

        json.dump(quality_report, f, indent=4, default=convert_to_serializable)

    print("Data quality report saved")

    print("SYSTEM COMPLETE")


if __name__ == "__main__":
    run()