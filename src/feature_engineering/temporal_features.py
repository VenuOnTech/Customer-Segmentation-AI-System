import pandas as pd

def add_temporal_features(df, mapping):

    customer_col = mapping["customer_id"]
    date_col = mapping["transaction_date"]

    # Ensure datetime
    df[date_col] = pd.to_datetime(df[date_col])

    # Sort data
    df = df.sort_values(by=[customer_col, date_col])

    # 🔹 Previous purchase date
    df["Prev_Date"] = df.groupby(customer_col)[date_col].shift(1)

    # 🔹 Days between purchases
    df["Days_Between"] = (df[date_col] - df["Prev_Date"]).dt.days

    # 🔹 Aggregate per customer
    temporal = df.groupby(customer_col).agg({
        "Days_Between": "mean",
        date_col: ["min", "max", "count"]
    })

    temporal.columns = [
        "Avg_Interval",
        "First_Purchase",
        "Last_Purchase",
        "Total_Purchases"
    ]

    # 🔹 Customer lifetime
    temporal["Customer_Lifetime"] = (
        (temporal["Last_Purchase"] - temporal["First_Purchase"]).dt.days + 1
    )

    # 🔹 Purchase velocity
    temporal["Purchase_Velocity"] = (
        temporal["Total_Purchases"] / temporal["Customer_Lifetime"]
    )

    # 🔹 Handle NaNs (single purchase customers)
    temporal["Avg_Interval"] = temporal["Avg_Interval"].fillna(0)

    # 🔹 Recency trend (basic version)
    temporal["Recency_Trend"] = temporal["Avg_Interval"]

    temporal = temporal.reset_index()

    return temporal