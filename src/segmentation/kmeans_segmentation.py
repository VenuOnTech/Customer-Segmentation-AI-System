from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from src.feature_engineering.feature_selection import select_features
from src.feature_engineering.feature_weighting import apply_feature_weights
from src.optimization.rl_optimizer import optimize_k_rl
import numpy as np


# ==========================================
# SAFE EVALUATION FUNCTION
# ==========================================
def evaluate_clustering(X, labels):
    try:
        # ❗ Prevent crash when only 1 cluster
        if len(set(labels)) < 2:
            return -1
        return silhouette_score(X, labels)
    except:
        return -1


# ==========================================
# MAIN FUNCTION
# ==========================================
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
    # K SELECTION (SAFE RL)
    # ==============================
    adaptive = config.get("clustering", {}).get("adaptive", False)
    mode = config.get("mode", "fast")

    if adaptive and mode == "full":
        print("🤖 Running RL optimization for K...")

        def evaluate_fn(k):
            try:
                model = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = model.fit_predict(X_scaled)
                return evaluate_clustering(X_scaled, labels)
            except:
                return -1

        try:
            # ✅ SAFE LIMIT (prevents infinite RL loop)
            n_clusters = optimize_k_rl(
                X_scaled,
                evaluate_fn,
                max_steps=10   # 🔥 IMPORTANT FIX
            )
            print(f"🤖 RL Optimized K: {n_clusters}")

        except Exception as e:
            print(f"⚠️ RL failed: {e}")
            n_clusters = config["clustering"].get("n_clusters", 3)

    else:
        n_clusters = config.get("clustering", {}).get("n_clusters", 3)

    # ==============================
    # KMEANS MODEL
    # ==============================
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=config["clustering"].get("random_state", 42),
        n_init=config["clustering"].get("n_init", 10)
    )

    kmeans_labels = kmeans.fit_predict(X_scaled)

    # ==============================
    # OPTIONAL DBSCAN (SAFE)
    # ==============================
    use_dbscan = config.get("clustering", {}).get("use_dbscan", False)

    if use_dbscan:
        print("🔍 Running DBSCAN...")
        dbscan = DBSCAN(eps=0.5, min_samples=5, n_jobs=-1)
        dbscan_labels = dbscan.fit_predict(X_scaled)

        # Hybrid logic
        final_labels = np.where(dbscan_labels == -1, kmeans_labels, dbscan_labels)

        rfm["DBSCAN_Cluster"] = dbscan_labels
    else:
        final_labels = kmeans_labels

    # ==============================
    # SAVE RESULTS
    # ==============================
    rfm["KMeans_Cluster"] = kmeans_labels
    rfm["Final_Cluster"] = final_labels
    rfm["Cluster"] = final_labels

    # ==============================
    # METRICS (SAFE)
    # ==============================
    try:
        if len(set(final_labels)) > 1:
            sil_score = silhouette_score(X_scaled, final_labels)
            print(f"🔥 Final Silhouette Score: {sil_score:.4f}")
        else:
            sil_score = None
            print("⚠️ Only one cluster detected")
    except:
        sil_score = None

    metrics = {
        "n_clusters": int(len(set(final_labels))),
        "silhouette_score": sil_score
    }

    return rfm, kmeans, scaler, metrics