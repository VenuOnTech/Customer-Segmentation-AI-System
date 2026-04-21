from src.data_ingestion.schema_contract import REQUIRED_SCHEMA

def detect_columns(df):
    """
    Detect required columns with strict validation
    """

    mapping = {}
    missing = []

    for key, possible_names in REQUIRED_SCHEMA.items():
        found = None

        for col in df.columns:
            if col.lower() in [p.lower() for p in possible_names]:
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