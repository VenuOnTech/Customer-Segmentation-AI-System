from sklearn.cluster import DBSCAN

def run_dbscan(rfm):
    model = DBSCAN(eps=0.5)
    rfm["DBSCAN_Cluster"] = model.fit_predict(rfm)
    return rfm