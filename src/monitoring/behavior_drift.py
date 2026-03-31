def detect_drift(old_mean, new_mean):

    change = ((new_mean - old_mean) / old_mean) * 100

    return change < -10