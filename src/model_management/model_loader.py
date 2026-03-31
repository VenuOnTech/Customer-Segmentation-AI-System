import joblib, json

def load_latest():

    with open("models/metadata.json") as f:
        data = json.load(f)

    v = data["latest_version"]

    base = f"models/v{v}"

    return (
        joblib.load(f"{base}/kmeans_model.pkl"),
        joblib.load(f"{base}/churn_model.pkl"),
        joblib.load(f"{base}/scaler.pkl")
    )