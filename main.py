from src.data_ingestion.load_data import load_data
from src.data_ingestion.schema_detection import detect_columns
from src.preprocessing.data_cleaning import clean_data
from src.feature_engineering.rfm_features import create_rfm
from src.feature_engineering.multi_source_features import add_multi_source_features
from src.segmentation.kmeans_segmentation import run_kmeans
from src.prediction.churn_prediction import train_churn
from src.prediction.future_prediction import predict_future_purchase
from src.explainability.shap_explainer import explain_customer
from src.monitoring.behavior_drift import detect_drift
from src.model_management.model_versioning import save_models

import os


def run():
    print("🚀 Starting pipeline...")

    # 🔹 Load Data (supports fallback if missing)
    df = load_data("data/raw/Online_Retail.xlsx")

    # 🔹 Detect Schema
    mapping = detect_columns(df)

    # 🔹 Add Features
    df = add_multi_source_features(df)

    # 🔹 Clean Data
    df = clean_data(df, mapping)

    # 🔹 RFM
    rfm = create_rfm(df, mapping)

    # 🔹 Segmentation
    rfm, kmeans, scaler = run_kmeans(rfm)

    # 🔹 Future Prediction
    rfm = predict_future_purchase(rfm)

    # 🔹 Churn Model
    churn_model = train_churn(rfm)

    # 🔹 Explainability
    rfm["Explanation"] = rfm.apply(explain_customer, axis=1)

    # 🔹 Drift Detection (basic safe logic)
    old_mean = rfm["Frequency"].mean()
    new_mean = old_mean * 0.9

    if detect_drift(old_mean, new_mean):
        print("⚠️ Drift detected → retraining suggested")

    # 🔥 Save models
    save_models(kmeans, churn_model, scaler)

    # 🔥 Save output
    os.makedirs("outputs", exist_ok=True)
    rfm.to_csv("outputs/customer_segments.csv", index=False)

    print("✅ Results saved to outputs/customer_segments.csv")
    print("🎉 PIPELINE COMPLETE")


if __name__ == "__main__":
    run()