from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

def compare_models(X):
    results = {}

    kmeans = KMeans(n_clusters=3, random_state=42)
    labels_k = kmeans.fit_predict(X)
    results["kmeans"] = silhouette_score(X, labels_k)

    agg = AgglomerativeClustering(n_clusters=3)
    labels_a = agg.fit_predict(X)
    results["agglomerative"] = silhouette_score(X, labels_a)

    best_model = max(results, key=results.get)

    print(f"🏆 Best model: {best_model}")
    return best_model, results