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

def run():

    df = load_data("data/raw/Online_Retail.xlsx")

    mapping = detect_columns(df)

    df = add_multi_source_features(df)

    df = clean_data(df, mapping)

    rfm = create_rfm(df, mapping)

    rfm, kmeans, scaler = run_kmeans(rfm)

    rfm = predict_future_purchase(rfm)

    churn = train_churn(rfm)

    rfm["Explanation"] = rfm.apply(explain_customer, axis=1)

    old_mean = rfm["Frequency"].mean()
    new_mean = old_mean * 0.8

    if detect_drift(old_mean, new_mean):
        print("Drift detected → retraining needed")

    print("SYSTEM COMPLETE")

if __name__ == "__main__":
    run()