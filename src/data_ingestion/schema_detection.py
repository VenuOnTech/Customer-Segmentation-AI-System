"""
Schema detection and column mapping for the Online_Retail dataset
"""

def detect_columns(df):
    """
    Map Online_Retail dataset columns to standard names.
<<<<<<< Updated upstream
    """
    return {
=======
    Returns a dictionary mapping standard names to actual column names.
    """
    
    mapping = {
>>>>>>> Stashed changes
        'customer_id': 'CustomerID',
        'transaction_date': 'InvoiceDate',
        'quantity': 'Quantity',
        'price': 'UnitPrice'
    }
<<<<<<< Updated upstream
=======
    
    # Validate that required columns exist
    required_cols = list(mapping.values())
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return mapping
>>>>>>> Stashed changes
