import random

def optimize_pipeline(config, drift=False):
    """
    Simple RL-like adaptive optimizer (safe, no infinite loops)
    """

    clustering_cfg = config.get("clustering", {})

    # 🔁 Adjust clusters dynamically
    if drift:
        clustering_cfg["n_clusters"] = min(
            clustering_cfg.get("n_clusters", 3) + 1,
            10
        )

    # 🎯 Adjust feature weights randomly (exploration)
    feature_weights = {
        "Recency": random.uniform(0.8, 1.2),
        "Frequency": random.uniform(0.8, 1.2),
        "Monetary": random.uniform(0.8, 1.2),
    }

    config["feature_weights"] = feature_weights

    return config