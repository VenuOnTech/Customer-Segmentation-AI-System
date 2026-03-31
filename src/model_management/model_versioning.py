import os
import json
import joblib

META_PATH = "models/metadata.json"

def get_new_version():
    if not os.path.exists(META_PATH):
        return 1

    with open(META_PATH) as f:
        data = json.load(f)

    return data.get("latest_version", 0) + 1


def save_models(kmeans, churn_model, scaler):

    version = get_new_version()
    version_path = f"models/v{version}"

    os.makedirs(version_path, exist_ok=True)

    joblib.dump(kmeans, f"{version_path}/kmeans_model.pkl")
    joblib.dump(churn_model, f"{version_path}/churn_model.pkl")
    joblib.dump(scaler, f"{version_path}/scaler.pkl")

    metadata = {
        "latest_version": version
    }

    with open(META_PATH, "w") as f:
        json.dump(metadata, f)

    print(f"Models saved in version: v{version}")