from scipy.stats import ks_2samp

def detect_drift(old_data, new_data):

    stat, p_value = ks_2samp(old_data, new_data)

    print(f"KS Statistic: {stat}")
    print(f"P-value: {p_value}")

    # If distributions differ significantly
    return p_value < 0.05