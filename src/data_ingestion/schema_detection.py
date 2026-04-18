def detect_columns(df):
    """
    Map Online_Retail dataset columns to standard names.
    """

    mapping = {
        'customer_id': 'CustomerID',
        'transaction_date': 'InvoiceDate',
        'quantity': 'Quantity',
        'price': 'UnitPrice'
    }

    required_cols = list(mapping.values())
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    return mapping