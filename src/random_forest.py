# src/random_forest.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Original RF on raw features (kept for backward compatibility).
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted", zero_division=0)
    rec = recall_score(y_test, preds, average="weighted", zero_division=0)
    f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

    print("\n📊 Random Forest Evaluation Metrics:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}\n")

    print("Detailed classification report:\n")
    print(classification_report(y_test, preds, zero_division=0))

    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Random Forest")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    return model


def train_hybrid_random_forest(encoder, autoencoder, X_train, y_train, X_test, y_test,
                               rf_params=None, hybrid_scaler_path=None,
                               verbose=True):
    """
    Train RandomForest on hybrid features: [latent Z || reconstruction_error].
    encoder: Keras encoder model (returns latent vector)
    autoencoder: full Keras autoencoder (for reconstructions)
    X_train/X_test: already scaled inputs (same scaler used to train AE)
    Returns: (rf_model, hybrid_scaler)
    """
    if rf_params is None:
        rf_params = {"n_estimators": 100, "random_state": 42, "n_jobs": -1}

    # Extract latent representations
    Z_train = encoder.predict(X_train, verbose=0)
    Z_test = encoder.predict(X_test, verbose=0)

    # Reconstruction error
    recon_train = autoencoder.predict(X_train, verbose=0)
    recon_test = autoencoder.predict(X_test, verbose=0)
    err_train = np.mean(np.square(X_train - recon_train), axis=1).reshape(-1, 1)
    err_test = np.mean(np.square(X_test - recon_test), axis=1).reshape(-1, 1)

    # Concatenate latent + error -> hybrid features
    hybrid_X_train = np.hstack([Z_train, err_train])
    hybrid_X_test = np.hstack([Z_test, err_test])

    # New scaler for hybrid feature space
    hybrid_scaler = StandardScaler()
    hybrid_X_train_scaled = hybrid_scaler.fit_transform(hybrid_X_train)
    hybrid_X_test_scaled = hybrid_scaler.transform(hybrid_X_test)

    # Save hybrid scaler if path provided
    if hybrid_scaler_path is not None:
        joblib.dump(hybrid_scaler, hybrid_scaler_path)
        if verbose:
            print(f"✅ Hybrid scaler saved at {hybrid_scaler_path}")

    # Train RF
    rf = RandomForestClassifier(**rf_params)
    rf.fit(hybrid_X_train_scaled, y_train)

    # Evaluate
    preds = rf.predict(hybrid_X_test_scaled)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted", zero_division=0)
    rec = recall_score(y_test, preds, average="weighted", zero_division=0)
    f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

    if verbose:
        print("\n📊 Hybrid Random Forest Evaluation Metrics:")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}\n")
        print(classification_report(y_test, preds, zero_division=0))

        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix - Hybrid RF")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.show()

    return rf, hybrid_scaler
