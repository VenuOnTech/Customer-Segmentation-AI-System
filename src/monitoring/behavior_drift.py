import os
import pandas as pd
from scipy.stats import ks_2samp

HISTORY_PATH = "outputs/feature_history.csv"

def detect_drift(new_data: pd.Series, feature_name="Frequency"):

    # If no history → save and skip drift
    if not os.path.exists(HISTORY_PATH):
        pd.DataFrame({feature_name: new_data}).to_csv(HISTORY_PATH, index=False)
        print("📦 No history found → baseline created")
        return False

    old_df = pd.read_csv(HISTORY_PATH)

    if feature_name not in old_df.columns:
        print("⚠️ Feature not found in history → skipping drift")
        return False

    old_data = old_df[feature_name]

    if len(old_data) < 10 or len(new_data) < 10:
        return False

    stat, p_value = ks_2samp(old_data, new_data)

    print(f"KS Statistic: {stat}")
    print(f"P-value: {p_value}")

    # Update history
    combined = pd.concat([old_df, pd.DataFrame({feature_name: new_data})])
    combined.to_csv(HISTORY_PATH, index=False)

    return p_value < 0.05