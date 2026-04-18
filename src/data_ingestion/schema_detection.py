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
    column_map = {}

    # Normalize column names
    cols = {col.lower(): col for col in df.columns}

    # Mapping logic
    column_map["customer_id"] = cols.get("customerid")
    column_map["invoice_date"] = cols.get("invoicedate")
    column_map["quantity"] = cols.get("quantity")
    column_map["price"] = cols.get("unitprice")

    # 🔹 Confidence calculation
    found = sum(v is not None for v in column_map.values())
    confidence = found / 4
    print(f"Schema detection confidence: {confidence:.2f}")

    # 🔹 STRICT VALIDATION (important fix)
    missing = [k for k, v in column_map.items() if v is None]

    if missing:
        print(f"❌ Missing required columns: {missing}")
        raise ValueError(f"Missing required columns: {missing}")

    return column_map