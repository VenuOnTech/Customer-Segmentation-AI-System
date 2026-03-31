import json

def detect_columns(df):
    with open("config/column_aliases.json") as f:
        aliases = json.load(f)

    mapping = {}

    for key, values in aliases.items():
        for col in df.columns:
            if col.lower() in values:
                mapping[key] = col

    return mapping