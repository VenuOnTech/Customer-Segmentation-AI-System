import pandas as pd
import os


def load_data(path):
    """Load data with fallback support"""
    try:
        if path.endswith(".csv"):
            return pd.read_csv(path)
        else:
            return pd.read_excel(path)

    except Exception as e:
        print(f"⚠️ Failed to load {path}: {e}")

        # fallback (for CI / GitHub Actions)
        fallback = "data/sample.csv"
        if os.path.exists(fallback):
            print("📦 Using fallback sample data")
            return pd.read_csv(fallback)

        raise FileNotFoundError("No data source available")