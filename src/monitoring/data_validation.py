def validate_data(df, mapping, strict=True):

    errors = []

    # Always check nulls
    if df[mapping["customer_id"]].isnull().any():
        errors.append("CustomerID contains null values")

    if strict:
        if (df[mapping["quantity"]] <= 0).any():
            errors.append("Quantity contains non-positive values")

        if (df[mapping["price"]] <= 0).any():
            errors.append("Price contains non-positive values")

    if errors:
        raise ValueError(f"Data validation failed: {errors}")

    print("✅ Data validation passed")