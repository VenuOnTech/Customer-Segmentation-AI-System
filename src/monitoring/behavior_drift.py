def detect_drift(old_rfm, new_rfm):

    change = (new_rfm.mean() - old_rfm.mean()) / old_rfm.mean() * 100

    if change["Frequency"] < -10:
        return True

    return False