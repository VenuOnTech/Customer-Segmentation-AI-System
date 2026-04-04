def detect_columns(df):
    return {col: 'numeric' if df[col].dtype in ['int64', 'float64'] else 'string' for col in df.columns}