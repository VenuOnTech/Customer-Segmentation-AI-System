import hashlib
import pandas as pd

def get_data_version(df: pd.DataFrame):
    """
    Generate a simple hash for dataset versioning
    """
    data_bytes = df.to_csv(index=False).encode()
    return hashlib.md5(data_bytes).hexdigest()