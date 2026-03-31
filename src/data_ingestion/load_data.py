import pandas as pd

def load_data(path):
    if path.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_excel(path)