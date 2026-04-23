def add_multi_source_features(df):

    df = df.copy()  # 🔥 FIX

    if "Review" in df.columns:
        df.loc[:, "Sentiment"] = df["Review"].astype(str).str.lower().apply(
            lambda x: 1 if "good" in x else 0
        )
    else:
        df.loc[:, "Sentiment"] = 0

    df.loc[:, "EngagementScore"] = df.get("Quantity", 1) * 0.1

    return df