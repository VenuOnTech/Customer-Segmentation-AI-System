"""
Data cleaning utilities
"""
import pandas as pd

def clean_data(df, mapping):
    """
    Clean the data by removing nulls and invalid rows
    
    Args:
        df: DataFrame to clean
        mapping: Column name mapping
    
    Returns:
        Cleaned DataFrame
    """
    
    # Remove rows with missing values in key columns
    key_cols = list(mapping.values())
    df_clean = df.dropna(subset=key_cols)
    
    # Remove rows with non-positive quantities
    quantity_col = mapping['quantity']
    df_clean = df_clean[df_clean[quantity_col] > 0]
    
    # Remove rows with non-positive prices
    price_col = mapping['price']
    df_clean = df_clean[df_clean[price_col] > 0]
    
    return df_clean.reset_index(drop=True)