def add_multi_source_features(df):

    if "Review" in df.columns:
        df["Sentiment"] = df["Review"].apply(
            lambda x: 1 if "good" in str(x).lower() else 0
        )
    else:
        df["Sentiment"] = 0

    df["EngagementScore"] = df.get("Quantity", 1) * 0.1

    return df