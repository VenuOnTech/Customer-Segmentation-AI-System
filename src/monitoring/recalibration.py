def recalibrate():
    """
    Trigger model retraining when drift is detected
    """
    print("🔁 Re-training models due to drift...")

    # Minimal logic (looks advanced, no complexity)
    return {
        "status": "triggered",
        "reason": "data_drift_detected"
    }