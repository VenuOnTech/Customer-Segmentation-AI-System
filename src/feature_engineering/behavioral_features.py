import pandas as pd

def add_behavioral_features(df, mapping):
    """
    Adds advanced behavioral features per customer
    """

    customer_col = mapping["customer_id"]
    date_col = mapping["transaction_date"]
    quantity_col = mapping["quantity"]
    price_col = mapping["price"]

    df[date_col] = pd.to_datetime(df[date_col])

    # 🔹 Sort data
    df = df.sort_values(by=[customer_col, date_col])

    # 🔹 Inter-purchase time
    df["Prev_Date"] = df.groupby(customer_col)[date_col].shift(1)
    df["Interpurchase_Time"] = (df[date_col] - df["Prev_Date"]).dt.days

    # 🔹 Customer-level aggregation
    behavioral = df.groupby(customer_col).agg({
        "Interpurchase_Time": ["mean", "std"],
        quantity_col: ["mean", "std"],
        price_col: ["mean", "std"]
    })

    # 🔹 Flatten column names
    behavioral.columns = [
        "Avg_Interpurchase_Time",
        "Std_Interpurchase_Time",
        "Avg_Quantity",
        "Std_Quantity",
        "Avg_Price",
        "Std_Price"
    ]

    behavioral = behavioral.reset_index()

    return behavioral