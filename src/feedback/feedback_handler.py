import os
import pandas as pd

def collect_feedback(feedback_path="data/feedback/real_feedback.csv"):

    if not os.path.exists(feedback_path):
        print("⚠️ No real feedback found")
        return None

    df = pd.read_csv(feedback_path)

    required_cols = ["CustomerID", "Actual_Churn"]

    if not all(col in df.columns for col in required_cols):
        print("⚠️ Invalid feedback format")
        return None

    print("📥 Real feedback loaded")
    return df


def retrain_with_feedback(model, X, y):
    print("🔁 Retraining model with feedback data...")

    model.fit(X, y)

    return model