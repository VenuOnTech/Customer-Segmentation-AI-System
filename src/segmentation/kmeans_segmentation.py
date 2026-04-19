from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from src.feature_engineering.feature_selection import select_features
from src.feature_engineering.feature_weighting import apply_feature_weights


def find_optimal_k(X_scaled, max_k=8):

    scores = {}

    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)

        score = silhouette_score(X_scaled, labels)
        scores[k] = score

        print(f"K={k}, Silhouette Score={score:.4f}")

    best_k = max(scores, key=scores.get)

    # 🔹 Business-safe override
    if best_k == 2 and scores.get(3, 0) > 0.55:
        best_k = 3

    return best_k


def run_kmeans(rfm, config):

    # 🔹 Feature selection (ONLY THIS — no select_dtypes)
    X = select_features(rfm)

    # 🔹 Remove label column if exists
    if "Cluster" in X.columns:
        X = X.drop(columns=["Cluster"])

    # 🔹 Handle missing values
    X = X.fillna(0)
    
    X = apply_feature_weights(X)

    # 🔹 Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 🔹 Adaptive K
    if config["clustering"].get("adaptive", False):
        n_clusters = find_optimal_k(X_scaled)
        print(f"✅ Adaptive K selected: {n_clusters}")
    else:
        n_clusters = config["clustering"]["n_clusters"]

    model = KMeans(
        n_clusters=n_clusters,
        random_state=config["clustering"]["random_state"],
        n_init=config["clustering"]["n_init"]
    )

    rfm["Cluster"] = model.fit_predict(X_scaled)

    return rfm, model, scaler