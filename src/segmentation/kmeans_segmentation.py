from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from src.feature_engineering.feature_selection import select_features
from src.feature_engineering.feature_weighting import apply_feature_weights
from src.optimization.rl_optimizer import optimize_k_rl
import numpy as np


def evaluate_clustering(X, labels):
    try:
        if len(set(labels)) < 2:
            return -1
        return silhouette_score(X, labels)
    except:
        return -1


def run_kmeans(rfm, config):

    X = select_features(rfm)
    X = X.select_dtypes(include=["number"]).copy()
    X = X.replace([np.inf, -np.inf], 0).fillna(0)

    X = apply_feature_weights(X, config)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    adaptive = config.get("clustering", {}).get("adaptive", False)
    mode = config.get("mode", "lite")

    # ==========================================
    # SAFE RL
    # ==========================================
    if adaptive and mode == "full":

        def evaluate_fn(k):
            try:
                model = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = model.fit_predict(X_scaled)
                return evaluate_clustering(X_scaled, labels)
            except:
                return -1

        try:
            n_clusters = optimize_k_rl(
                X_scaled,
                evaluate_fn,
                max_steps=10  # 🔥 prevents infinite loop
            )
            print(f"🤖 RL Optimized K: {n_clusters}")

        except Exception as e:
            print(f"⚠️ RL failed: {e}")
            n_clusters = 3
    else:
        n_clusters = config.get("clustering", {}).get("n_clusters", 3)

    # ==========================================
    # KMEANS
    # ==========================================
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    kmeans_labels = kmeans.fit_predict(X_scaled)

    # ==========================================
    # OPTIONAL DBSCAN
    # ==========================================
    use_dbscan = config.get("clustering", {}).get("use_dbscan", False)

    if use_dbscan:
        dbscan = DBSCAN(eps=0.5, min_samples=5, n_jobs=-1)
        dbscan_labels = dbscan.fit_predict(X_scaled)

        final_labels = np.where(dbscan_labels == -1, kmeans_labels, dbscan_labels)
        rfm["DBSCAN_Cluster"] = dbscan_labels
    else:
        final_labels = kmeans_labels

    rfm["KMeans_Cluster"] = kmeans_labels
    rfm["Final_Cluster"] = final_labels
    rfm["Cluster"] = final_labels

    # ==========================================
    # METRICS
    # ==========================================
    try:
        if len(set(final_labels)) > 1:
            sil_score = silhouette_score(X_scaled, final_labels)
        else:
            sil_score = None
    except:
        sil_score = None

    metrics = {
        "n_clusters": int(len(set(final_labels))),
        "silhouette_score": sil_score
    }

    return rfm, kmeans, scaler, metrics