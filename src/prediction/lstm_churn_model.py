import numpy as np

# 🔥 SAFE IMPORT
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import MinMaxScaler
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


# ==============================
# TRAIN LSTM (BEHAVIOR MODEL)
# ==============================
def train_lstm_model(rfm):

    if not TF_AVAILABLE:
        print("⚠️ TensorFlow not available → skipping LSTM")
        return None, None

    features = ["Recency", "Frequency", "Monetary"]

    data = rfm[features].values

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = [], []

    # 🔥 Sequence creation (last 5 steps)
    for i in range(5, len(data_scaled)):
        X.append(data_scaled[i-5:i])
        y.append(data_scaled[i][1])  # predict frequency (behavior signal)

    X, y = np.array(X), np.array(y)

    if len(X) < 10:
        print("⚠️ Not enough data for LSTM")
        return None, None

    model = Sequential([
        LSTM(32, input_shape=(X.shape[1], X.shape[2])),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

    model.fit(X, y, epochs=3, batch_size=64, verbose=0)

    print("✅ LSTM model trained")

    return model, scaler


# ==============================
# PREDICT LSTM BEHAVIOR
# ==============================
def predict_lstm(model, scaler, rfm):

    if model is None or scaler is None:
        rfm["LSTM_Score"] = 0.0
        return rfm

    features = ["Recency", "Frequency", "Monetary"]

    data = scaler.transform(rfm[features].values)

    preds = []

    for i in range(len(data)):
        if i < 5:
            preds.append(0.0)
        else:
            seq = data[i-5:i]
            pred = model.predict(seq.reshape(1, 5, 3), verbose=0)[0][0]
            preds.append(float(pred))

    rfm["LSTM_Score"] = preds

    return rfm