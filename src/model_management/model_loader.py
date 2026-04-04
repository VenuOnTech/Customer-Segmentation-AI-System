import os
import json
import joblib

META_PATH = "models/metadata.json"

def load_latest():
    """Load the latest version of saved models."""
    
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Model metadata not found at {META_PATH}. Run main.py first to train models.")
    
    with open(META_PATH) as f:
        metadata = json.load(f)
    
    latest_version = metadata.get("latest_version", 0)
    
    if latest_version == 0:
        raise ValueError("No models have been trained yet. Run main.py to train models.")
    
    version_path = f"models/v{latest_version}"
    
    if not os.path.exists(version_path):
        raise FileNotFoundError(f"Model version {latest_version} not found at {version_path}")
    
    # Load models
    kmeans = joblib.load(f"{version_path}/kmeans_model.pkl")
    churn_model = joblib.load(f"{version_path}/churn_model.pkl")
    scaler = joblib.load(f"{version_path}/scaler.pkl")
    
    print(f"✅ Loaded models from version: v{latest_version}")
    
    return kmeans, churn_model, scaler, latest_version