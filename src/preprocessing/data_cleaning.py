def clean_data(df, mapping):

    df_clean = df.copy()

    print(f"Initial row count: {len(df_clean)}")

    key_cols = list(mapping.values())
    df_clean = df_clean.dropna(subset=key_cols)

    quantity_col = mapping['quantity']
    df_clean = df_clean[df_clean[quantity_col] > 0]

    price_col = mapping['price']
    df_clean = df_clean[df_clean[price_col] > 0]

    df_clean = df_clean.drop_duplicates()

    print(f"Final row count: {len(df_clean)}")

    return df_clean.reset_index(drop=True)