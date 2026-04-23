# 🔥 SAFE IMPORT
try:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def generate_autoencoder_features(df):

    if not TF_AVAILABLE:
        print("⚠️ TensorFlow not available → skipping autoencoder")
        return df

    numeric_df = df.select_dtypes(include=["number"]).fillna(0)

    scaler = MinMaxScaler()
    data = scaler.fit_transform(numeric_df)

    input_dim = data.shape[1]

    input_layer = Input(shape=(input_dim,))
    encoded = Dense(16, activation="relu")(input_layer)
    encoded = Dense(8, activation="relu")(encoded)

    decoded = Dense(16, activation="relu")(encoded)
    decoded = Dense(input_dim, activation="sigmoid")(decoded)

    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)

    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.fit(data, data, epochs=5, verbose=0)

    encoded_features = encoder.predict(data, verbose=0)

    encoded_df = pd.DataFrame(
        encoded_features,
        columns=[f"AE_{i}" for i in range(encoded_features.shape[1])]
    )

    df = pd.concat([df.reset_index(drop=True), encoded_df], axis=1)

    return df