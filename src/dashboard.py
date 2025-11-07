# src/dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import time
from sklearn.metrics import confusion_matrix, f1_score, precision_score, accuracy_score

from realtime_capture import capture_packet, predict_packet  # local import

# -------------------- Paths -------------------- #
MODELS_DIR = os.path.join("data", "processed", "models")
AUTO_PATH = os.path.join(MODELS_DIR, "autoencoder.h5")             # or saved model dir if you changed
ENCODER_PATH = os.path.join(MODELS_DIR, "encoder.h5")
RF_PATH = os.path.join(MODELS_DIR, "random_forest.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
HYBRID_SCALER_PATH = os.path.join(MODELS_DIR, "hybrid_scaler.pkl")
AE_THRESHOLD_PATH = os.path.join(MODELS_DIR, "ae_threshold.pkl")

# -------------------- Load Models -------------------- #
def load_models():
    models = {}
    # Autoencoder (store path)
    if os.path.exists(AUTO_PATH):
        models["Autoencoder"] = {"auto_path": AUTO_PATH}
    # Random Forest (load object)
    if os.path.exists(RF_PATH):
        try:
            rf = joblib.load(RF_PATH)
            models["Random Forest"] = {"rf": rf}
        except Exception as e:
            st.warning(f"Failed to load Random Forest: {e}")
    # Hybrid (require auto + encoder + rf)
    if os.path.exists(AUTO_PATH) and os.path.exists(ENCODER_PATH) and os.path.exists(RF_PATH):
        try:
            models["Hybrid"] = {
                "auto_path": AUTO_PATH,
                "encoder_path": ENCODER_PATH,
                "rf": joblib.load(RF_PATH),
                "hybrid_scaler": joblib.load(HYBRID_SCALER_PATH) if os.path.exists(HYBRID_SCALER_PATH) else None
            }
        except Exception as e:
            st.warning(f"Failed to load Hybrid components: {e}")
    return models


# -------------------- Load Scaler & Threshold -------------------- #
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
ae_threshold = joblib.load(AE_THRESHOLD_PATH) if os.path.exists(AE_THRESHOLD_PATH) else 0.01

# -------------------- Streamlit UI -------------------- #
st.set_page_config(page_title="NetMorph Realtime Dashboard", layout="wide")
st.title("🌐 NetMorph Realtime Network Analysis Dashboard")

# Sidebar Controls
st.sidebar.header("⚙️ Controls")
interface = st.sidebar.text_input("Network Interface", value="en0")
packet_count = st.sidebar.number_input("Packets per batch", min_value=1, max_value=100, value=20)
refresh_interval = st.sidebar.number_input("Refresh Interval (s)", min_value=1, max_value=60, value=5)

# Load models
models = load_models()
if not models:
    st.error("❌ No trained models found in `data/processed/models`")
    st.stop()

selected_model = st.sidebar.selectbox("Select Model", list(models.keys()))
start_capture = st.sidebar.button("🚀 Start Realtime Capture")
simulate_from_csv = st.sidebar.checkbox("Simulate from CSV (no live sniff)", value=True)
csv_file = None
if simulate_from_csv:
    csv_file = st.sidebar.file_uploader("Upload features CSV (optional)", type=["csv"])

if start_capture:
    st.success(f"Realtime capture started using **{selected_model}**")

    all_packets = []
    all_predictions = []

    # Placeholders for Streamlit outputs
    df_placeholder = st.empty()
    summary_placeholder = st.empty()
    cm_placeholder = st.empty()

    try:
        # If simulating from CSV, load samples into a list and iterate instead of sniffing
        sim_rows = None
        sim_idx = 0
        if simulate_from_csv:
            # load dataframe from uploader or default file(s)
            if csv_file is not None:
                df_sim = pd.read_csv(csv_file)
            else:
                default_path = os.path.join("data", "processed", "features", "all_features.csv")
                if os.path.exists(default_path):
                    df_sim = pd.read_csv(default_path)
                else:
                    feat_dir = os.path.join("data", "processed", "features")
                    csvs = [f for f in os.listdir(feat_dir) if f.endswith(".csv")]
                    if not csvs:
                        st.error("No feature CSV found for simulation.")
                        st.stop()
                    df_sim = pd.read_csv(os.path.join(feat_dir, csvs[0]))

            # Clean column names (strip whitespace)
            df_sim.columns = df_sim.columns.str.strip()

            # Drop label column if present
            if "Label" in df_sim.columns:
                df_sim = df_sim.drop(columns=["Label"])

            # If we have a scaler from training, try to select/reorder scaler columns
            if scaler is not None:
                feat_names = getattr(scaler, "feature_names_in_", None)
                if feat_names is not None:
                    missing = [c for c in feat_names if c not in df_sim.columns]
                    if missing:
                        st.warning(f"CSV is missing scaler columns: {missing[:5]}... Falling back to numeric-only selection.")
                    else:
                        df_sim = df_sim.loc[:, feat_names]  # reorder/select exact columns
                else:
                    # fallback to numeric columns only
                    df_sim = df_sim.select_dtypes(include=[np.number]).copy()
                    for c in df_sim.columns:
                        df_sim[c] = pd.to_numeric(df_sim[c], errors="coerce")
            else:
                # no scaler: keep numeric columns only
                df_sim = df_sim.select_dtypes(include=[np.number]).copy()
                for c in df_sim.columns:
                    df_sim[c] = pd.to_numeric(df_sim[c], errors="coerce")

            # Drop columns that became all-NaN after coercion
            df_sim = df_sim.dropna(axis=1, how="all")

            if df_sim.shape[1] == 0:
                st.error("No numeric feature columns found in CSV after cleaning.")
                st.stop()

            X_sim = df_sim.values
            sim_rows = [row for row in X_sim]

            st.info(f"Loaded simulation CSV with shape {X_sim.shape}.")

        while True:
            # get packets (either from capture or simulation)
            if simulate_from_csv and sim_rows is not None:
                # slice the next batch
                packets = []
                for _ in range(packet_count):
                    if sim_idx >= len(sim_rows):
                        sim_idx = 0  # loop
                    packets.append(sim_rows[sim_idx])
                    sim_idx += 1
            else:
                packets = capture_packet(interface=interface, count=packet_count)

            batch_predictions = []

            for pkt in packets:
                # Ensure features is 2D for scaler/predict; attacker vector shapes handled below
                features = np.array(pkt).reshape(1, -1)

                # debug info
                st.write("features raw shape:", features.shape)

                # Apply input scaler (used by AE training) if present
                if scaler is not None:
                    try:
                        features = scaler.transform(features)
                    except Exception as e:
                        st.warning(f"Scaler transform failed: {e}")

                # Dispatch depending on selected model
                if selected_model == "Autoencoder":
                    auto_path = models[selected_model]["auto_path"]
                    label = predict_packet(
                        features.flatten(),
                        model=auto_path,
                        model_type="Autoencoder",
                        threshold=ae_threshold
                    )

                elif selected_model == "Random Forest":
                    rf_model = models[selected_model]["rf"]

                    # show RF expectations (helpful debugging in Streamlit)
                    st.write(f"RF expects {getattr(rf_model,'n_features_in_', 'unknown')} features; RF classes: {getattr(rf_model,'classes_', None)}")

                    # Attempt to pull hybrid artifacts if available (so predict_packet can build hybrid vector)
                    hybrid_meta = models.get("Hybrid", {})
                    encoder_path = hybrid_meta.get("encoder_path", None)
                    auto_path = hybrid_meta.get("auto_path", None)
                    hybrid_scaler = hybrid_meta.get("hybrid_scaler", None)

                    label = predict_packet(
                        features.flatten(),
                        model=auto_path,
                        model_type="Random Forest",
                        rf_model=rf_model,
                        encoder=encoder_path,
                        hybrid_scaler=hybrid_scaler
                    )

                elif selected_model == "Hybrid":
                    hybrid_meta = models["Hybrid"]
                    auto_path = hybrid_meta["auto_path"]
                    encoder_path = hybrid_meta["encoder_path"]
                    rf_model = hybrid_meta["rf"]
                    hybrid_scaler = hybrid_meta.get("hybrid_scaler", None)

                    label = predict_packet(
                        features.flatten(),
                        model=auto_path,
                        model_type="Hybrid",
                        encoder=encoder_path,
                        hybrid_scaler=hybrid_scaler,
                        rf_model=rf_model,
                        threshold=ae_threshold
                    )
                else:
                    label = 0

                batch_predictions.append(label)

            all_packets.extend(packets)
            all_predictions.extend(batch_predictions)

            # ---------------- Live Packet Predictions ---------------- #
            df = pd.DataFrame({
                "Packet": range(1, len(all_predictions) + 1),
                "Prediction": ["Malicious" if x==1 else "Benign" for x in all_predictions]
            })
            df_placeholder.dataframe(df.tail(20), use_container_width=True)

            # ---------------- Metrics Summary ---------------- #
            if len(all_predictions) > 1:
                # Dummy true labels (replace with actual if available)
                y_true = [0]*len(all_predictions)
                y_pred = all_predictions

                acc = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                prec = precision_score(y_true, y_pred, zero_division=0)

                summary_placeholder.markdown(f"""
                **Metrics (approximate)**  
                Accuracy: {acc:.2f}  
                F1 Score: {f1:.2f}  
                Precision: {prec:.2f}
                """)

                # ---------------- Confusion Matrix ---------------- #
                cm = confusion_matrix(y_true, y_pred)
                plt.figure(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                cm_placeholder.pyplot(plt)
                plt.clf()

            time.sleep(refresh_interval)

    except KeyboardInterrupt:
        st.warning("Realtime capture stopped manually.")
