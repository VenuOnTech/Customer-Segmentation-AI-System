def predict_future_purchase(df):

    df["Purchase_Probability"] = (
        (1 / (1 + df["Recency"])) * 0.5 +
        (df["Frequency"] / (df["Frequency"].max() + 1)) * 0.3 +
        (df["Monetary"] / (df["Monetary"].max() + 1)) * 0.2
    )

    return df