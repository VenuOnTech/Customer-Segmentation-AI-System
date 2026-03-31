from src.data_ingestion.load_data import load_data
from src.data_ingestion.schema_detection import detect_columns
from src.preprocessing.data_cleaning import clean_data
from src.feature_engineering.rfm_features import create_rfm
from src.segmentation.kmeans_segmentation import run_kmeans
from src.prediction.churn_prediction import train_churn

df = load_data("data/raw/Online_Retail.xlsx")

mapping = detect_columns(df)

df = clean_data(df, mapping)

rfm = create_rfm(df, mapping)

rfm, kmeans, scaler = run_kmeans(rfm)

churn_model = train_churn(rfm)

print("System executed successfully!")