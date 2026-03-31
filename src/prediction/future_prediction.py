def predict_future_purchase(rfm):

    rfm["Purchase_Probability"] = (
        rfm["Frequency"] / (rfm["Recency"] + 1)
    )

    return rfm