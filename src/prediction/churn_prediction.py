from sklearn.ensemble import RandomForestClassifier


def train_churn(rfm):

    rfm["Churn"] = (rfm["Recency"] > 90).astype(int)

    X = rfm[["Recency", "Frequency", "Monetary", "Cluster"]]
    y = rfm["Churn"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model