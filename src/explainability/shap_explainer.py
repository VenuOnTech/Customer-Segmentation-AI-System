"""
SHAP Explainer for customer segmentation
"""

def explain_customer(row):
    """
    Generate explanations for customer behavior based on RFM metrics.
    
    Args:
        row: DataFrame row with customer RFM values
        
    Returns:
        String with explanation reasons
    """
    
    reasons = []
    
    # Check recency (days since last purchase)
    if row["Recency"] > 90:
        reasons.append("Inactive customer (>90 days)")
    elif row["Recency"] > 60:
        reasons.append("Low activity (>60 days)")
    
    # Check frequency (purchase count)
    if row["Frequency"] < 2:
        reasons.append("Low purchase frequency")
    elif row["Frequency"] < 5:
        reasons.append("Moderate frequency")
    
    # Check monetary (total spending)
    if row["Monetary"] < 50:
        reasons.append("Low spending (<$50)")
    elif row["Monetary"] < 100:
        reasons.append("Low spending (<$100)")
    
    # No reasons = good customer
    if not reasons:
        reasons.append("Loyal, active customer")
    
    return ", ".join(reasons)