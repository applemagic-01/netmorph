# src/realtime_capture.py
import numpy as np
from scapy.all import sniff
from tensorflow.keras.models import load_model
import joblib
import os

def capture_packet(interface="en0", count=10):
    """
    Capture `count` packets from `interface` using scapy.sniff and return
    a list of fixed-length numeric feature vectors (float arrays).
    """
    packets_list = []

    def process_packet(pkt):
        pkt_features = [
            len(pkt),
            getattr(pkt, "sport", 0),
            getattr(pkt, "dport", 0),
            getattr(pkt, "proto", 0)
        ]
        # pad to expected length (78) to match training shape
        while len(pkt_features) < 78:
            pkt_features.append(0.0)
        packets_list.append(np.array(pkt_features, dtype=float))

    sniff(iface=interface, prn=process_packet, count=count, store=False)
    return packets_list


def predict_packet(pkt_features, model=None, model_type="Autoencoder",
                   threshold=0.01, encoder=None, hybrid_scaler=None, rf_model=None):
    """
    Unified prediction interface.

    pkt_features: 1D array-like (already scaled if AE expects scaled input)
    model: for Autoencoder/Hybrid -> path or loaded autoencoder
    encoder: encoder path or loaded encoder (used for Hybrid or hybrid-RF inference)
    hybrid_scaler: sklearn scaler used for latent+error (if hybrid)
    rf_model: sklearn RandomForest model instance (required for RF/Hybrid)
    """
    features = np.array(pkt_features).reshape(1, -1)

    # ---------- Autoencoder-only ----------
    if model_type == "Autoencoder":
        if isinstance(model, str):
            auto = load_model(model, compile=False)
        else:
            auto = model
        reconstructed = auto.predict(features, verbose=0)
        mse = np.mean(np.power(features - reconstructed, 2))
        return 1 if mse > threshold else 0

    # ---------- RandomForest-only ----------
    if model_type == "Random Forest":
        if rf_model is None:
            raise ValueError("rf_model is required for Random Forest inference")

        expected = getattr(rf_model, "n_features_in_", None)

        # If RF expects same number as provided, use features directly
        if expected is None or expected == features.shape[1]:
            X_for_rf = features
        else:
            # RF expects different dimensionality -> try to construct hybrid vector
            if encoder is None or model is None or hybrid_scaler is None:
                raise ValueError(
                    f"RF expects {expected} features but input has {features.shape[1]}. "
                    "Provide encoder, autoencoder and hybrid_scaler to create hybrid features."
                )
            # load encoder/auto if paths provided
            if isinstance(encoder, str):
                enc = load_model(encoder, compile=False)
            else:
                enc = encoder
            if isinstance(model, str):
                auto = load_model(model, compile=False)
            else:
                auto = model

            # If AE was trained on scaled inputs, it's expected that features passed here are already scaled.
            latent = enc.predict(features, verbose=0)                 # (1, latent_dim)
            reconstructed = auto.predict(features, verbose=0)
            mse = np.mean(np.power(features - reconstructed, 2), axis=1).reshape(-1, 1)
            hybrid_vec = np.hstack([latent, mse])                    # (1, latent_dim+1)
            X_for_rf = hybrid_scaler.transform(hybrid_vec)

        pred = rf_model.predict(X_for_rf)[0]

        # handle predicted label types: numeric or string
        try:
            return int(pred)
        except Exception:
            if isinstance(pred, str):
                label_str = pred.strip().upper()
                if label_str in ("BENIGN", "NORMAL", "0", "FALSE"):
                    return 0
                return 1
            try:
                return int(np.round(float(pred)))
            except Exception:
                return 1 if str(pred).strip().upper() != "BENIGN" else 0

    # ---------- Hybrid (explicit) ----------
    if model_type == "Hybrid":
        if encoder is None or rf_model is None or model is None:
            raise ValueError("encoder, autoencoder (model) and rf_model are required for Hybrid inference")

        if isinstance(encoder, str):
            enc = load_model(encoder, compile=False)
        else:
            enc = encoder
        if isinstance(model, str):
            auto = load_model(model, compile=False)
        else:
            auto = model

        latent = enc.predict(features, verbose=0)
        reconstructed = auto.predict(features, verbose=0)
        mse = np.mean(np.power(features - reconstructed, 2), axis=1).reshape(-1, 1)
        hybrid_vec = np.hstack([latent, mse])

        if hybrid_scaler is not None:
            hybrid_vec = hybrid_scaler.transform(hybrid_vec)

        pred = rf_model.predict(hybrid_vec)[0]
        try:
            return int(pred)
        except Exception:
            if isinstance(pred, str):
                return 0 if pred.strip().upper() in ("BENIGN", "NORMAL", "0") else 1
            try:
                return int(np.round(float(pred)))
            except Exception:
                return 1 if str(pred).strip().upper() != "BENIGN" else 0

    raise ValueError(f"Unknown model_type: {model_type}")
