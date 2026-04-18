import pandas as pd

def load_data(path):
    """Load data from CSV or Excel"""
    return pd.read_csv(path) if path.endswith('.csv') else pd.read_excel(path)

def detect_columns(df):
    """Detect schema"""
    return {
        col: 'numeric' if df[col].dtype in ['int64', 'float64'] else 'string'
        for col in df.columns
    }