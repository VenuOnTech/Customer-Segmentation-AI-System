import os

def save_features(df):
    try:
        # ✅ Correct path inside outputs
        save_dir = "outputs/feature_store"
        os.makedirs(save_dir, exist_ok=True)

        file_path = os.path.join(save_dir, "rfm_features_v1.csv")

        df.to_csv(file_path, index=False)

        print(f"📦 Feature store saved at: {file_path}")

    except Exception as e:
        print(f"❌ Failed to save feature store: {e}")