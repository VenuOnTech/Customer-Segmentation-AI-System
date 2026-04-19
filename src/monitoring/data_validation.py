def validate_data(df, mapping, strict=True):

    errors = []
    warnings = []

    # --- NULL CHECKS ---
    if df[mapping["customer_id"]].isnull().any():
        msg = "CustomerID contains null values"
        if strict:
            errors.append(msg)
        else:
            warnings.append(msg)

    if df[mapping["price"]].isnull().any():
        msg = "Price contains null values"
        if strict:
            errors.append(msg)
        else:
            warnings.append(msg)

    # --- VALUE CHECKS ---
    if strict:
        if (df[mapping["quantity"]] <= 0).any():
            errors.append("Quantity contains non-positive values")

        if (df[mapping["price"]] <= 0).any():
            errors.append("Price contains non-positive values")

    # --- OUTPUT ---
    if warnings:
        print(f"⚠️ Data warnings: {warnings}")

    if errors:
        raise ValueError(f"Data validation failed: {errors}")

    print("✅ Data validation passed")