from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def run_kmeans(rfm):

    scaler = StandardScaler()
    X = scaler.fit_transform(rfm)

    model = KMeans(n_clusters=4)
    rfm["Cluster"] = model.fit_predict(X)

    return rfm, model, scaler