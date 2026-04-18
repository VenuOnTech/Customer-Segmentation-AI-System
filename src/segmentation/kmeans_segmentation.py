from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def find_optimal_k(X_scaled, max_k=8):

    best_k = 2
    best_score = -1

    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)

        score = silhouette_score(X_scaled, labels)

        if score > best_score:
            best_score = score
            best_k = k

    return best_k


def run_kmeans(rfm, config):

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    numeric_cols = ['Recency', 'Frequency', 'Monetary']
    X = rfm[numeric_cols].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ✅ Adaptive K
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