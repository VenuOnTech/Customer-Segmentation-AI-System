"""
Data cleaning utilities
"""
import pandas as pd

def clean_data(df, mapping):
    """
    Clean the data by removing nulls and invalid rows.
    
    Args:
        df: DataFrame to clean
        mapping: Column name mapping
    
    Returns:
        Cleaned DataFrame
    """
    
    # Make a copy to avoid modifying original
    df_clean = df.copy()
    
    print(f"Initial row count: {len(df_clean)}")
    
    # Remove rows with missing values in key columns
    key_cols = list(mapping.values())
    df_clean = df_clean.dropna(subset=key_cols)
    print(f"After removing nulls: {len(df_clean)} rows")
    
    # Remove rows with non-positive quantities
    quantity_col = mapping['quantity']
    df_clean = df_clean[df_clean[quantity_col] > 0]
    print(f"After removing non-positive quantities: {len(df_clean)} rows")
    
    # Remove rows with non-positive prices
    price_col = mapping['price']
    df_clean = df_clean[df_clean[price_col] > 0]
    print(f"After removing non-positive prices: {len(df_clean)} rows")
    
    # Remove duplicate transactions
    df_clean = df_clean.drop_duplicates()
    print(f"After removing duplicates: {len(df_clean)} rows")
    
    return df_clean.reset_index(drop=True)