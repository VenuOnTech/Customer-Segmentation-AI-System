from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def run_kmeans(rfm, k=4):

    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm)

    model = KMeans(n_clusters=k, random_state=42)
    rfm["Cluster"] = model.fit_predict(scaled)

    return rfm, model, scaler