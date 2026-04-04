def detect_columns(df):
    """
    Map Online_Retail dataset columns to standard names.
    """
    return {
        'customer_id': 'CustomerID',
        'transaction_date': 'InvoiceDate',
        'quantity': 'Quantity',
        'price': 'UnitPrice'
    }
