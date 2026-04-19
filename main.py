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

import json
import numpy as np
import os


def run():

    # ✅ ALWAYS create outputs folder first
    os.makedirs("outputs", exist_ok=True)

    config = load_config()

    # 🔹 Ensure output directory exists FIRST
    os.makedirs("outputs", exist_ok=True)

    # 🔹 Load Data
    df = load_data("data/raw/Online_Retail.xlsx")
<<<<<<< Updated upstream
=======
<<<<<<< HEAD

    # 🔹 Save snapshot (after directory exists)
    df.sample(min(1000, len(df))).to_csv("outputs/data_snapshot.csv", index=False)
=======
>>>>>>> 519fad9c0b5123d5f9372fd277bf3b2c8e440dc1
>>>>>>> Stashed changes

    # 🔹 Detect Schema
    mapping = detect_columns(df)

    # 🔹 Soft Validation (before cleaning)
    validate_data(df, mapping, strict=False)

    # 🔹 Add Multi-source Features
    df = add_multi_source_features(df)

    # 🔹 Clean Data
    df = clean_data(df, mapping)

<<<<<<< Updated upstream
=======
<<<<<<< HEAD
    # 🔹 Strict Validation (after cleaning)
    validate_data(df, mapping, strict=True)
=======
>>>>>>> Stashed changes
    # 🔹 Validate Data (EARLY STOP if bad)
    validate_data(df, mapping)

    # 🔹 Data Quality Report (on CLEAN data)
    quality_report = generate_data_quality_report(df)
    with open("outputs/data_quality_report.json", "w") as f:
        json.dump(quality_report, f, indent=4)
    print("Data quality report saved")

    # 🔹 Save Data Snapshot (CLEAN data)
    df.sample(min(1000, len(df))).to_csv("outputs/data_snapshot.csv", index=False)
<<<<<<< Updated upstream
=======
>>>>>>> 519fad9c0b5123d5f9372fd277bf3b2c8e440dc1
>>>>>>> Stashed changes

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

    # 🔹 REAL Drift Detection (historical comparison)
    previous_path = "outputs/previous_frequency.npy"
    current_data = rfm["Frequency"].values

    if os.path.exists(previous_path):
        old_data = np.load(previous_path)

<<<<<<< Updated upstream
=======
<<<<<<< HEAD
=======
>>>>>>> Stashed changes
        if detect_drift(old_data, current_data):
            print("Drift detected → retraining needed")
    else:
        print("No previous data → skipping drift detection")

    np.save(previous_path, current_data)

<<<<<<< Updated upstream
=======
>>>>>>> 519fad9c0b5123d5f9372fd277bf3b2c8e440dc1
>>>>>>> Stashed changes
    # 🔥 SAVE MODELS
    save_models(kmeans, churn_model, scaler)

    # 🔥 SAVE OUTPUTS
    output_path = "outputs/customer_segments.csv"
    rfm.to_csv(output_path, index=True)

    print("Results saved to outputs/customer_segments.csv")

    # 🔁 FEEDBACK LOOP
    feedback_df = collect_feedback(output_path)

    if feedback_df is not None and not feedback_df.empty:
        X_feedback = feedback_df[["Recency", "Frequency", "Monetary"]]
        y_feedback = feedback_df["Actual_Churn"]

<<<<<<< Updated upstream
=======
<<<<<<< HEAD
    churn_model = retrain_with_feedback(churn_model, X_feedback, y_feedback)

    # 🔹 Data Quality Report
    quality_report = generate_data_quality_report(df)

    with open("outputs/data_quality_report.json", "w") as f:
        json.dump(quality_report, f, indent=4)

    print("Data quality report saved")
=======
>>>>>>> Stashed changes
        churn_model = retrain_with_feedback(churn_model, X_feedback, y_feedback)
        print("Model retrained with feedback")
    else:
        print("No feedback data available")
<<<<<<< Updated upstream
=======
>>>>>>> 519fad9c0b5123d5f9372fd277bf3b2c8e440dc1
>>>>>>> Stashed changes

    print("SYSTEM COMPLETE")


if __name__ == "__main__":
    run()
