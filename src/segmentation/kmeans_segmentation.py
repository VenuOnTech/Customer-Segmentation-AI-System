from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from src.feature_engineering.feature_selection import select_features
from src.feature_engineering.feature_weighting import apply_feature_weights
import numpy as np


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

    if best_k == 2 and scores.get(3, 0) > 0.55:
        best_k = 3

    return best_k


def run_kmeans(rfm, config):

    # 🔹 Feature selection
    X = select_features(rfm)

    # 🔹 Limit features (stability)
    MAX_FEATURES = 10
    if X.shape[1] > MAX_FEATURES:
        X = X.iloc[:, :MAX_FEATURES]

    # 🔹 Clean data
    X = X.select_dtypes(include=["number"]).copy()
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    if "Cluster" in X.columns:
        X = X.drop(columns=["Cluster"])

    # 🔹 Apply feature weights
    X = apply_feature_weights(X)

    # 🔹 Sampling (CI stability)
    MAX_ROWS = 100000
    if len(X) > MAX_ROWS:
        print(f"⚠️ Sampling data: {len(X)} → {MAX_ROWS}")
        X_sample = X.sample(MAX_ROWS, random_state=42)
    else:
        X_sample = X

    # 🔹 Scaling
    scaler = StandardScaler()
    X_scaled_sample = scaler.fit_transform(X_sample)

    # ==============================
    # 🔥 KMEANS
    # ==============================

    if config["clustering"].get("adaptive", False):
        n_clusters = find_optimal_k(X_scaled_sample)
        print(f"✅ Adaptive K: {n_clusters}")
    else:
        n_clusters = config["clustering"]["n_clusters"]

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=config["clustering"]["random_state"],
        n_init=config["clustering"]["n_init"],
        algorithm="lloyd"
    )

    kmeans.fit(X_scaled_sample)

    # ==============================
    # 🔥 DBSCAN (NEW)
    # ==============================

    dbscan = DBSCAN(
        eps=0.5,
        min_samples=5,
        n_jobs=-1
    )

    dbscan_labels_sample = dbscan.fit_predict(X_scaled_sample)

    # ==============================
    # 🔥 FULL DATA PREDICTION
    # ==============================

    X_scaled_full = scaler.transform(X)

    rfm["KMeans_Cluster"] = kmeans.predict(X_scaled_full)

    dbscan_full = DBSCAN(
        eps=0.5,
        min_samples=5,
        n_jobs=-1
    )

    rfm["DBSCAN_Cluster"] = dbscan_full.fit_predict(X_scaled_full)

    # ==============================
    # 🔥 HYBRID CLUSTER LOGIC
    # ==============================

    def combine_clusters(row):
        if row["DBSCAN_Cluster"] == -1:
            return row["KMeans_Cluster"]
        else:
            return row["DBSCAN_Cluster"]

    rfm["Final_Cluster"] = rfm.apply(combine_clusters, axis=1)

    return rfm, kmeans, scaler