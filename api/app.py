from fastapi import FastAPI
import pandas as pd
from src.model_management.model_loader import load_latest

app = FastAPI()

kmeans, churn_model, scaler, version = load_latest()


@app.get("/")
def home():
    return {"status": "API running", "model_version": version}


@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    prob = churn_model.predict_proba(df)[0][1]

    return {"churn_probability": float(prob)}


@app.post("/segment")
def segment(data: dict):

    df = pd.DataFrame([data])
    scaled = scaler.transform(df)

    cluster = kmeans.predict(scaled)[0]

    return {"cluster": int(cluster)}