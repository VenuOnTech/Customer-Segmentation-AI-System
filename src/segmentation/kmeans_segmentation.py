from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from src.feature_engineering.feature_selection import select_features
from src.feature_engineering.feature_weighting import apply_feature_weights
from sklearn.metrics import silhouette_score


def find_optimal_k(X_scaled, max_k=8):

    scores = {}

    for k in range(2, max_k + 1):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
            algorithm="lloyd"
        )

        labels = model.fit_predict(X_scaled)

        score = silhouette_score(X_scaled, labels)
        scores[k] = score

        print(f"K={k}, Silhouette Score={score:.4f}")

    best_k = max(scores, key=scores.get)

    # Avoid trivial clustering
    if best_k == 2 and scores.get(3, 0) > 0.55:
        best_k = 3

    return best_k

def run_kmeans(rfm, config):

    X = select_features(rfm)

    MAX_FEATURES = 10
    if X.shape[1] > MAX_FEATURES:
        X = X.iloc[:, :MAX_FEATURES]

    X = X.select_dtypes(include=["number"]).copy()
    X = X.replace([float("inf"), float("-inf")], 0)
    X = X.fillna(0)

    if "Cluster" in X.columns:
        X = X.drop(columns=["Cluster"])

    MAX_ROWS = 100000
    if len(X) > MAX_ROWS:
        print(f"⚠️ Sampling data for clustering: {len(X)} → {MAX_ROWS}")
        X_sample = X.sample(MAX_ROWS, random_state=42)
    else:
        X_sample = X

    scaler = StandardScaler()
    X_scaled_sample = scaler.fit_transform(X_sample)

    # ✅ FIX: use SAMPLE consistently
    if config["clustering"].get("adaptive", False):
        n_clusters = find_optimal_k(X_scaled_sample)
        print(f"✅ Adaptive K selected: {n_clusters}")
    else:
        n_clusters = config["clustering"]["n_clusters"]

    model = KMeans(
        n_clusters=n_clusters,
        random_state=config["clustering"]["random_state"],
        n_init=config["clustering"]["n_init"],
        algorithm="lloyd"
    )

    model.fit(X_scaled_sample)

    # 🔥 SAFE FULL TRANSFORM
    X_scaled_full = scaler.transform(X)
    rfm["Cluster"] = model.predict(X_scaled_full)

    return rfm, model, scaler