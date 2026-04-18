from sklearn.ensemble import RandomForestClassifier

def train_churn(rfm):

    # Define churn (target)
    rfm["Churn"] = (rfm["Recency"] > 90).astype(int)

    # ❗ FIX: Remove Cluster (leakage) & avoid direct dependency
    X = rfm[["Frequency", "Monetary"]]  
    y = rfm["Churn"]

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    return model