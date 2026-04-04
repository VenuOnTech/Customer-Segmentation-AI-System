"""
RFM (Recency, Frequency, Monetary) feature engineering
"""
import datetime as dt
import pandas as pd

def create_rfm(df, mapping):
    """
    Create RFM features from transaction data
    
    Args:
        df: DataFrame with transaction data
        mapping: Column name mapping dictionary
    
    Returns:
        DataFrame with RFM features
    """
    
    # Get column names from mapping
    customer_col = mapping['customer_id']
    date_col = mapping['transaction_date']
    qty_col = mapping['quantity']
    price_col = mapping['price']
    
    # Ensure date column is datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Create total column
    df['Total'] = df[price_col] * df[qty_col]
    
    # Calculate snapshot date
    snapshot = df[date_col].max() + dt.timedelta(days=1)
    
    # Create RFM
    rfm = df.groupby(customer_col).agg({
        date_col: lambda x: (snapshot - x.max()).days,  # Recency
        customer_col: 'count',  # Frequency
        'Total': 'sum',  # Monetary
    })
    
    # Rename columns
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    
    # Reset index to make customer_id a column
    rfm = rfm.reset_index()
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    
    return rfm