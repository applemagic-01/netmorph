# src/autoencoder.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def build_autoencoder_and_encoder(input_dim, latent_dim=16):
    """
    Build a functional autoencoder and a separate encoder model.
    latent_dim should match the bottleneck size you want (default 16).
    """
    input_layer = layers.Input(shape=(input_dim,), name="ae_input")
    x = layers.Dense(64, activation="relu")(input_layer)
    x = layers.Dense(32, activation="relu")(x)
    latent = layers.Dense(latent_dim, activation="relu", name="latent")(x)

    # Decoder
    x = layers.Dense(32, activation="relu")(latent)
    x = layers.Dense(64, activation="relu")(x)
    output_layer = layers.Dense(input_dim, activation="sigmoid", name="ae_output")(x)

    autoencoder = models.Model(inputs=input_layer, outputs=output_layer, name="autoencoder")
    encoder = models.Model(inputs=input_layer, outputs=latent, name="encoder")

    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder, encoder


def train_autoencoder(X_train, epochs=20, batch_size=64, latent_dim=16):
    """
    Train the autoencoder and return (autoencoder_model, encoder_model, history)
    """
    autoencoder, encoder = build_autoencoder_and_encoder(X_train.shape[1], latent_dim=latent_dim)
    history = autoencoder.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )
    return autoencoder, encoder, history


def compute_reconstruction_error(autoencoder, X):
    """
    Compute per-sample MSE reconstruction errors for X using the autoencoder.
    """
    reconstructions = autoencoder.predict(X, verbose=0)
    errors = np.mean(np.square(X - reconstructions), axis=1)
    return errors
