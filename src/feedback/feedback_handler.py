import pandas as pd

def collect_feedback(predictions_path="outputs/customer_segments.csv"):
    df = pd.read_csv(predictions_path)

    # Simulated feedback (you can later replace with real user input)
    df["Actual_Churn"] = df["Churn"]  # simulate ground truth

    df.to_csv("outputs/feedback_data.csv", index=False)
    return df


def retrain_with_feedback(model, X, y):
    print("🔁 Retraining model with feedback data...")
    model.fit(X, y)
    return model