import numpy as np
import json
import os

# 🔒 Prevent memory crashes
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

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
from src.feedback.feedback_handler import collect_feedback, retrain_with_feedback
from src.monitoring.data_quality import generate_data_quality_report
from src.monitoring.data_validation import validate_data
from src.data_ingestion.data_versioning import get_data_version
from src.monitoring.data_lineage import log_data_lineage
from src.feature_engineering.temporal_features import add_temporal_features
from src.feature_engineering.behavioral_features import add_behavioral_features
from src.explainability.feature_importance_explainer import generate_feature_importance_explanations
from src.prediction.lstm_churn_model import train_lstm_churn, predict_lstm

def run():

    config = load_config()

    os.makedirs("outputs", exist_ok=True)

    df = load_data("data/raw/Online_Retail.xlsx")
    df.sample(min(1000, len(df))).to_csv("outputs/data_snapshot.csv", index=False)

    mapping = detect_columns(df)

    validate_data(df, mapping, strict=False)

    df = add_multi_source_features(df)
    df = clean_data(df, mapping)

    data_version = get_data_version(df)
    print(f"Data Version: {data_version}")

    validate_data(df, mapping, strict=True)

    # ==============================
    # 🔥 FEATURE ENGINEERING
    # ==============================

    temporal_features = add_temporal_features(df, mapping)
    rfm = create_rfm(df, mapping)
    behavioral = add_behavioral_features(df, mapping)

    rfm = rfm.merge(behavioral, on=mapping["customer_id"], how="left")
    rfm = rfm.merge(temporal_features, on=mapping["customer_id"], how="left")

    rfm = rfm.fillna(0)

    numeric_cols = rfm.select_dtypes(include=["number"]).columns
    rfm[numeric_cols] = rfm[numeric_cols].astype("float32")

    # ==============================
    # 🔹 SEGMENTATION (HYBRID)
    # ==============================

    rfm, kmeans, scaler = run_kmeans(rfm, config)
    
    print("✅ Hybrid clustering applied (KMeans + DBSCAN)")

    # ==============================
    # 🔹 PREDICTION (DEEP MODEL)
    # ==============================

    rfm = predict_future_purchase(rfm)

    # 🔹 Traditional Model
    churn_model, churn_metrics = train_deep_churn(rfm)

    # 🔹 LSTM Model (SAFE MODE)
    lstm_model, lstm_metrics = train_lstm_churn(rfm)

    if lstm_model is not None:
        rfm["LSTM_Churn_Prob"] = predict_lstm(lstm_model, rfm)
        print("✅ LSTM model applied")
    else:
        rfm["LSTM_Churn_Prob"] = 0
        print("⚠️ LSTM skipped (TensorFlow not available)")

    churn_metrics.update(lstm_metrics)

    # ==============================
    # 🔍 EXPLAINABILITY (SAFE MODE)
    # ==============================

    EXPLAIN_SAMPLE_SIZE = 1000

    if len(rfm) > EXPLAIN_SAMPLE_SIZE:
        print(f"⚠️ Sampling for explanations: {len(rfm)} → {EXPLAIN_SAMPLE_SIZE}")
        explain_df = rfm.sample(EXPLAIN_SAMPLE_SIZE, random_state=42)
    else:
        explain_df = rfm

    X_explain = explain_df[["Frequency", "Monetary"]]

    explanations = generate_feature_importance_explanations(
        churn_model,
        X_explain
    )

    rfm["Explanation"] = "Not computed"
    rfm.loc[explain_df.index, "Explanation"] = explanations

    print("✅ Feature importance explanations generated")

    # ==============================
    # 🔹 DRIFT DETECTION
    # ==============================

    old_data = rfm["Frequency"]
    noise = np.random.normal(0, 0.01, len(rfm))
    new_data = rfm["Frequency"] * (1 + noise)

    if detect_drift(old_data, new_data):
        print("Drift detected → retraining needed")

    # ==============================
    # 🔥 SAVE
    # ==============================

    save_models(kmeans, churn_model, scaler)

    output_path = "outputs/customer_segments.csv"
    rfm.to_csv(output_path, index=True)

    print("Results saved")

    log_data_lineage(data_version, output_path)

    # ==============================
    # 🔁 FEEDBACK LOOP
    # ==============================

    feedback_df = collect_feedback(output_path)

    if feedback_df is not None and not feedback_df.empty:

        required_cols = ["Recency", "Frequency", "Monetary", "Actual_Churn"]

        if all(col in feedback_df.columns for col in required_cols):

            X_feedback = feedback_df[["Recency", "Frequency", "Monetary"]]
            y_feedback = feedback_df["Actual_Churn"]

            churn_model = retrain_with_feedback(churn_model, X_feedback, y_feedback)

            print("✅ Feedback retraining completed")

        else:
            print("⚠️ Missing columns in feedback")

    else:
        print("⚠️ No feedback data")

    # ==============================
    # 🔹 SAVE METRICS
    # ==============================

    with open("outputs/model_performance.json", "w") as f:

        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            return str(obj)

        json.dump(churn_metrics, f, indent=4, default=convert)

    print("Model performance saved")

    # ==============================
    # 🔹 DATA QUALITY
    # ==============================

    quality_report = generate_data_quality_report(df)

    with open("outputs/data_quality_report.json", "w") as f:
        json.dump(quality_report, f, indent=4, default=convert)

    print("SYSTEM COMPLETE")


if __name__ == "__main__":
    run()