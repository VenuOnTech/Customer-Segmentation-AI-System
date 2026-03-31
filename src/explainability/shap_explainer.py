def explain_customer(row):

    reasons = []

    if row["Recency"] > 60:
        reasons.append("Inactive customer")

    if row["Frequency"] < 2:
        reasons.append("Low purchase frequency")

    if row["Monetary"] < 100:
        reasons.append("Low spending")

    return ", ".join(reasons)