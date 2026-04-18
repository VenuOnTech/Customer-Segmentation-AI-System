def validate_schema(mapping):
    required = ["customer_id", "invoice_date", "quantity", "price"]

    missing = [col for col in required if col not in mapping.values()]

    if missing:
        print(f"⚠️ Missing required columns: {missing}")
        return False

    return True


def get_fallback_mapping(df):
    print("⚠️ Using fallback schema mapping")

    fallback = {
        "customer_id": "CustomerID",
        "invoice_date": "InvoiceDate",
        "quantity": "Quantity",
        "price": "UnitPrice"
    }

    return fallback


def detect_columns(df):
    mapping = {}
    confidence = 0

    for col in df.columns:
        col_lower = col.lower()

        if "customer" in col_lower:
            mapping["customer_id"] = col
            confidence += 1
        elif "date" in col_lower:
            mapping["invoice_date"] = col
            confidence += 1
        elif "quantity" in col_lower:
            mapping["quantity"] = col
            confidence += 1
        elif "price" in col_lower:
            mapping["price"] = col
            confidence += 1

    confidence_score = confidence / 4

    print(f"Schema detection confidence: {confidence_score:.2f}")

    if confidence_score < 0.75:
        print("⚠️ Low confidence in schema detection")

    if not validate_schema(mapping):
        mapping = get_fallback_mapping(df)

    return mapping