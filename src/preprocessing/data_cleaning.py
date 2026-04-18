def clean_data(df, mapping):

    initial_rows = len(df)

    # Track reasons
    null_customer = df[mapping["customer_id"]].isnull().sum()
    null_price = df[mapping["price"]].isnull().sum()
    negative_quantity = (df[mapping["quantity"]] <= 0).sum()

    print(f"Initial row count: {initial_rows}")
    print(f"Dropping null CustomerID rows: {null_customer}")
    print(f"Dropping null Price rows: {null_price}")
    print(f"Dropping negative Quantity rows: {negative_quantity}")

    # Apply cleaning
    df = df.dropna(subset=[mapping["customer_id"], mapping["price"]])
    df = df[df[mapping["quantity"]] > 0]

    final_rows = len(df)

    print(f"Final row count: {final_rows}")
    print(f"Total rows removed: {initial_rows - final_rows}")

    return df