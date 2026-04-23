from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def generate_autoencoder_features(df):

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
    autoencoder.fit(data, data, epochs=10, verbose=0)

    encoded_features = encoder.predict(data)

    encoded_df = pd.DataFrame(
        encoded_features,
        columns=[f"AE_{i}" for i in range(encoded_features.shape[1])]
    )

    df = pd.concat([df.reset_index(drop=True), encoded_df], axis=1)

    return df