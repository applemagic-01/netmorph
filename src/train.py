# src/train.py
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from src.features import load_all_features, preprocess_features
from src.autoencoder import train_autoencoder, compute_reconstruction_error
from src.random_forest import train_hybrid_random_forest
from sklearn.metrics import accuracy_score
import numpy as np

# Model paths (consistent with your repo)
MODELS_DIR = os.path.join("data", "processed", "models")
os.makedirs(MODELS_DIR, exist_ok=True)
AUTO_PATH = os.path.join(MODELS_DIR, "autoencoder.h5")
ENCODER_PATH = os.path.join(MODELS_DIR, "encoder.h5")
RF_PATH = os.path.join(MODELS_DIR, "random_forest.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
HYBRID_SCALER_PATH = os.path.join(MODELS_DIR, "hybrid_scaler.pkl")
AE_THRESHOLD_PATH = os.path.join(MODELS_DIR, "ae_threshold.pkl")

def main():
    # Load and preprocess
    df = load_all_features()
    X, y, scaler = preprocess_features(df, label_col="Label")

    # Persist input scaler for runtime
    joblib.dump(scaler, SCALER_PATH)
    print(f"✅ Scaler saved at {SCALER_PATH}")

    # train/test split (preserve your stratify)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Train Autoencoder (same method you used; returns encoder too)
    print("\n🔹 Training Autoencoder...")
    auto_model, encoder, _ = train_autoencoder(X_train, epochs=20, batch_size=128, latent_dim=16)

    # Compute AE threshold using benign samples from training (recommended)
    print("\n🔹 Computing Autoencoder Threshold...")
    train_errors = compute_reconstruction_error(auto_model, X_train)
    # derive threshold based on 'BENIGN' in y_train if labels are strings
    try:
        benign_mask = (y_train == "BENIGN")
    except Exception:
        benign_mask = (np.array(y_train) == 0)  # fallback if numeric labels
    if benign_mask.sum() == 0:
        # fallback to overall mean+2*std
        ae_threshold = np.mean(train_errors) + 2 * np.std(train_errors)
    else:
        ae_threshold = train_errors[benign_mask].mean() + 2 * train_errors[benign_mask].std()
    print(f"➡️ AE threshold: {ae_threshold:.8f}")

    # Save autoencoder, encoder, threshold
    auto_model.save(AUTO_PATH)
    encoder.save(ENCODER_PATH)
    joblib.dump(ae_threshold, AE_THRESHOLD_PATH)
    print(f"✅ Autoencoder saved at {AUTO_PATH}")
    print(f"✅ Encoder saved at {ENCODER_PATH}")
    print(f"✅ AE threshold saved at {AE_THRESHOLD_PATH}")

    # Train Hybrid RandomForest
    print("\n🔹 Training Hybrid Random Forest...")
    rf_model, hybrid_scaler = train_hybrid_random_forest(
        encoder, auto_model, X_train, y_train, X_test, y_test,
        rf_params=None, hybrid_scaler_path=HYBRID_SCALER_PATH
    )

    # Save RF and hybrid scaler
    joblib.dump(rf_model, RF_PATH)
    joblib.dump(hybrid_scaler, HYBRID_SCALER_PATH)
    print(f"✅ Hybrid Random Forest saved at {RF_PATH}")
    print(f"✅ Hybrid scaler saved at {HYBRID_SCALER_PATH}")

    # Print quick summary
    rf_acc = accuracy_score(y_test, rf_model.predict(
        hybrid_scaler.transform(
            np.hstack([encoder.predict(X_test, verbose=0),
                       np.mean(np.square(X_test - auto_model.predict(X_test, verbose=0)), axis=1).reshape(-1,1)
                      ])
        )
    ))
    print("\n📊 Training Complete!")
    print(f"Autoencoder AE-threshold: {ae_threshold:.6f}")
    print(f"Hybrid Random Forest Accuracy (test): {rf_acc:.4f}")

if __name__ == "__main__":
    main()
