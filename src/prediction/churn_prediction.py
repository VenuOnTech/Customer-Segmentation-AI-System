from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


def train_churn(rfm):

    # 🔹 Define features (ONLY numeric + useful)
    feature_cols = [
        col for col in rfm.columns
        if col not in ["CustomerID", "Cluster", "Churn", "Explanation"]
    ]

    X = rfm[feature_cols].select_dtypes(include=["number"]).fillna(0)

    # 🔹 Create target (simple heuristic if not present)
    if "Churn" not in rfm.columns:
        # Example logic: low frequency + high recency = churn
        rfm["Churn"] = ((rfm["Frequency"] < 2) & (rfm["Recency"] > 100)).astype(int)

    y = rfm["Churn"]

    # 🔹 Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 🔹 Model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=1
    )

    model.fit(X_train, y_train)

    # 🔹 Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 🔹 Metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }

    print("📊 Churn Model Performance:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return model, metrics