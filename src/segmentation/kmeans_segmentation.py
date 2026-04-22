from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from src.feature_engineering.feature_selection import select_features
from src.feature_engineering.feature_weighting import apply_feature_weights
from src.optimization.rl_optimizer import optimize_k_rl
import numpy as np


# ✅ FIX: Define evaluation function (MISSING BEFORE)
def evaluate_clustering(X, labels):
    try:
        return silhouette_score(X, labels)
    except:
        return -1


def find_optimal_k(X_scaled, max_k=8):
    scores = {}

    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)

        scores[k] = score
        print(f"K={k}, Silhouette Score={score:.4f}")

    best_k = max(scores, key=scores.get)
    return best_k


def run_kmeans(rfm, config):

    # ==============================
    # FEATURE PREPARATION
    # ==============================
    X = select_features(rfm)

    X = X.select_dtypes(include=["number"]).copy()
    X = X.replace([np.inf, -np.inf], 0).fillna(0)

    X = apply_feature_weights(X)

    # ==============================
    # SCALING
    # ==============================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ==============================
    # K SELECTION (FIXED RL BLOCK)
    # ==============================
    if config["clustering"].get("adaptive", False):

        def evaluate_fn(k):
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = model.fit_predict(X_scaled)
            return evaluate_clustering(X_scaled, labels)

        n_clusters = optimize_k_rl(X_scaled, evaluate_fn)
        print(f"🤖 RL Optimized K: {n_clusters}")

    else:
        n_clusters = config["clustering"]["n_clusters"]

    # ==============================
    # KMEANS MODEL
    # ==============================
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=config["clustering"]["random_state"],
        n_init=config["clustering"]["n_init"]
    )

    kmeans_labels = kmeans.fit_predict(X_scaled)

    # ==============================
    # DBSCAN MODEL
    # ==============================
    dbscan = DBSCAN(eps=0.5, min_samples=5, n_jobs=-1)
    dbscan_labels = dbscan.fit_predict(X_scaled)

    # ==============================
    # HYBRID CLUSTERING
    # ==============================
    final_labels = np.where(dbscan_labels == -1, kmeans_labels, dbscan_labels)

    # ==============================
    # SAVE RESULTS
    # ==============================
    rfm["KMeans_Cluster"] = kmeans_labels
    rfm["DBSCAN_Cluster"] = dbscan_labels
    rfm["Cluster"] = final_labels
    rfm["Final_Cluster"] = final_labels

    # ==============================
    # METRICS
    # ==============================
    try:
        sil_score = silhouette_score(X_scaled, final_labels)
        print(f"🔥 Final Silhouette Score: {sil_score:.4f}")
    except:
        sil_score = None

    metrics = {
        "n_clusters": int(len(set(final_labels))),
        "silhouette_score": sil_score
    }

    return rfm, kmeans, scaler, metrics