import os
import pandas as pd

def load_all_datasets(data_dir="data/raw"):

    dfs = []

    for file in os.listdir(data_dir):
        path = os.path.join(data_dir, file)

        try:
            if file.endswith(".csv"):
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)

            dfs.append(df)

        except Exception as e:
            print(f"⚠️ Skipping file {file}: {e}")
            continue

    if not dfs:
        raise ValueError("No datasets found")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"📊 Loaded {len(dfs)} datasets")

    return combined