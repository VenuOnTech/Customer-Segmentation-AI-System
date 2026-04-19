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


from src.data_ingestion.schema_contract import REQUIRED_SCHEMA

def detect_columns(df):

    mapping = {}
    missing = []

    for key, possible_names in REQUIRED_SCHEMA.items():
        found = None
        for col in df.columns:
            if col in possible_names:
                found = col
                break
        
        if found:
            mapping[key] = found
        else:
            missing.append(key)

    confidence = 1 - (len(missing) / len(REQUIRED_SCHEMA))
    print(f"Schema detection confidence: {confidence:.2f}")

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return mapping

    # 🔹 Robust mapping
    column_map["customer_id"] = find_col(["customerid", "customer_id"])
    column_map["transaction_date"] = find_col(["invoicedate", "invoice_date", "date"])
    column_map["quantity"] = find_col(["quantity", "qty"])
    column_map["price"] = find_col(["unitprice", "price"])

    # 🔹 Confidence
    found = sum(v is not None for v in column_map.values())
    confidence = found / 4
    print(f"Schema detection confidence: {confidence:.2f}")

    # 🔹 Strict validation
    missing = [k for k, v in column_map.items() if v is None]

    if missing:
        print(f"❌ Missing required columns: {missing}")
        raise ValueError(f"Missing required columns: {missing}")

    return column_map