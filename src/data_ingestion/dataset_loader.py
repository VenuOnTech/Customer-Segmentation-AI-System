import os
import pandas as pd

def load_all_datasets(data_dir="data/raw"):

    dfs = []

    for file in os.listdir(data_dir):
        if file.endswith(".csv") or file.endswith(".xlsx"):
            path = os.path.join(data_dir, file)

            if file.endswith(".csv"):
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)

            df["__source_file"] = file
            dfs.append(df)

    if not dfs:
        raise ValueError("No datasets found")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"📊 Loaded {len(dfs)} datasets")

    return combined