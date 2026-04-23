import os
import pandas as pd

FEATURE_STORE_PATH = "feature_store/"

def save_features(df, name="rfm_features"):

    os.makedirs(FEATURE_STORE_PATH, exist_ok=True)

    version = len(os.listdir(FEATURE_STORE_PATH)) + 1
    path = f"{FEATURE_STORE_PATH}/{name}_v{version}.csv"

    df.to_csv(path, index=False)

    print(f"💾 Features saved: {path}")

    return path