import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam


def train_lstm_churn(rfm):

    # 🔹 Features (can expand later)
    feature_cols = ["Recency", "Frequency", "Monetary"]

    X = rfm[feature_cols].values
    y = rfm["Churn"].values

    # 🔹 Normalize manually (safe)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # 🔹 Reshape for LSTM → (samples, timesteps, features)
    X = X.reshape((X.shape[0], 1, X.shape[1]))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 🔹 Build LSTM model
    model = Sequential([
        LSTM(32, input_shape=(X.shape[1], X.shape[2])),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # 🔹 Train (LOW epochs → CI safe)
    model.fit(
        X_train,
        y_train,
        epochs=3,
        batch_size=64,
        verbose=0
    )

    # 🔹 Evaluate
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)

    print(f"LSTM Accuracy: {acc:.4f}")

    return model, {"lstm_accuracy": acc}


def predict_lstm(model, rfm):

    feature_cols = ["Recency", "Frequency", "Monetary"]

    X = rfm[feature_cols].values
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    X = X.reshape((X.shape[0], 1, X.shape[1]))

    preds = model.predict(X)
    return preds.flatten()