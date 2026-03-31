import datetime as dt

def create_rfm(df, mapping):

    df["Total"] = df[mapping["price"]] * df[mapping["quantity"]]

    snapshot = df[mapping["transaction_date"]].max() + dt.timedelta(days=1)

    rfm = df.groupby(mapping["customer_id"]).agg({
        mapping["transaction_date"]: lambda x: (snapshot - x.max()).days,
        mapping["customer_id"]: "count",
        "Total": "sum",
        "Sentiment": "mean",
        "EngagementScore": "mean"
    })

    rfm.columns = ["Recency","Frequency","Monetary","Sentiment","Engagement"]

    return rfm