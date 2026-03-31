from sklearn.ensemble import RandomForestClassifier

def train_churn(rfm):

    rfm["Churn"] = (rfm["Recency"] > 90).astype(int)

    X = rfm[["Recency","Frequency","Monetary","Cluster"]]
    y = rfm["Churn"]

    model = RandomForestClassifier()
    model.fit(X,y)

    return model