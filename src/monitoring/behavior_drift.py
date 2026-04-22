from scipy.stats import ks_2samp

def detect_drift(old_data, new_data):

    # Prevent meaningless comparison
    if len(old_data) < 10 or len(new_data) < 10:
        return False

    stat, p_value = ks_2samp(old_data, new_data)

    print(f"KS Statistic: {stat}")
    print(f"P-value: {p_value}")

    return p_value < 0.05