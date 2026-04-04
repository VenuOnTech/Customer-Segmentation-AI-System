from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def run_kmeans(rfm):
    """
    Run K-Means clustering on RFM features.
    
    Args:
        rfm: DataFrame with RFM features
        
    Returns:
        rfm: DataFrame with Cluster column added
        model: Trained KMeans model
        scaler: StandardScaler fitted on RFM features
    """
    
    # Select only numeric columns for clustering
    numeric_cols = ['Recency', 'Frequency', 'Monetary']
    X = rfm[numeric_cols].copy()
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train KMeans
    model = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm["Cluster"] = model.fit_predict(X_scaled)
    
    return rfm, model, scaler