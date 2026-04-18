import numpy as np

def detect_drift(old_series, new_series):

    old_mean = np.mean(old_series)
    new_mean = np.mean(new_series)

    if old_mean == 0:
        return False

    change = ((new_mean - old_mean) / old_mean) * 100

    print(f"Drift change: {round(change,2)}%")

    return abs(change) > 10