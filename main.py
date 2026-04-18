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

import numpy as np
import os


def run():

    config = load_config()
    
    # 🔹 Load Data
    df = load_data("data/raw/Online_Retail.xlsx")

    # 🔹 Detect Schema
    mapping = detect_columns(df)

    # 🔹 Add Multi-source Features
    df = add_multi_source_features(df)

    # 🔹 Clean Data
    df = clean_data(df, mapping)

    # 🔹 Create RFM
    rfm = create_rfm(df, mapping)

    # 🔹 Segmentation
    rfm, kmeans, scaler = run_kmeans(rfm, config)

    # 🔹 Future Prediction
    rfm = predict_future_purchase(rfm)

    # 🔹 Churn Model
    churn_model = train_churn(rfm)

    # 🔹 Explainability
    X_explain = rfm[["Frequency", "Monetary"]]
    rfm["Explanation"] = generate_shap_explanations(churn_model, X_explain)

    # 🔹 Drift Detection
    old_data = rfm["Frequency"]
    noise = np.random.normal(0, 0.01, len(rfm))
    new_data = rfm["Frequency"] * (1 + noise)

    if detect_drift(old_data, new_data):
        print("Drift detected → retraining needed")

    # 🔥 SAVE MODELS (VERSIONING)
    save_models(kmeans, churn_model, scaler)

    # 🔥 SAVE OUTPUTS
    os.makedirs("outputs", exist_ok=True)
    output_path = "outputs/customer_segments.csv"
    rfm.to_csv(output_path, index=True)

    print("Results saved to outputs/customer_segments.csv")

    # 🔁 FEEDBACK LOOP
    feedback_df = collect_feedback(output_path)

    X_feedback = feedback_df[["Recency", "Frequency", "Monetary"]]
    y_feedback = feedback_df["Actual_Churn"]

    churn_model = retrain_with_feedback(churn_model, X_feedback, y_feedback)

    print("SYSTEM COMPLETE")


if __name__ == "__main__":
    run()