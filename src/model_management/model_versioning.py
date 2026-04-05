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

    # Preserve history
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            metadata = json.load(f)
    else:
        metadata = {}

    metadata["latest_version"] = version
    metadata[f"v{version}"] = {"status": "saved"}

    os.makedirs("models", exist_ok=True)

    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"✅ Models saved in version: v{version}")