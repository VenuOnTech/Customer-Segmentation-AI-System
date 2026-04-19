from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_deep_churn(rfm):

    X = rfm.select_dtypes(include=["number"]).drop(columns=["Cluster"], errors="ignore")
    
    # Fake target if not present (for now)
    if "Churn" not in rfm.columns:
        rfm["Churn"] = (rfm["Frequency"] < 2).astype(int)

    y = rfm["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=50)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    print(f"Deep Model Accuracy: {acc:.4f}")

    return model, {"accuracy": acc}