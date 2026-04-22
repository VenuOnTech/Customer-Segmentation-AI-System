import pandas as pd

def collect_feedback(predictions_path="outputs/customer_segments.csv"):
    df = pd.read_csv(predictions_path)

    # Simulated real feedback
    df["Actual_Churn"] = df["Churn"]

    df.to_csv("outputs/feedback_data.csv", index=False)

    print("📥 Feedback collected")

    return df


def retrain_with_feedback(model, X, y):
    print("🔁 Retraining model with feedback data...")

    model.fit(X, y)

    return model