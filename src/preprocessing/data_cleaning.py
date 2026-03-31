import pandas as pd

def clean_data(df, mapping):
    df = df.dropna(subset=[mapping["customer_id"]])
    df = df.drop_duplicates()

    df[mapping["transaction_date"]] = pd.to_datetime(df[mapping["transaction_date"]])

    return df