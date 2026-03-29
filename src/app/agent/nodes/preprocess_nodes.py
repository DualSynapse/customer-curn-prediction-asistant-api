# src/app/agent/nodes/preprocess_nodes.py

import joblib
import pandas as pd
from pathlib import Path
from loguru import logger
from src.app.agent.schema import AgentState

# ── Path ke file model pipeline ──────────────────────────────────────────────
# Path dihitung dari lokasi file ini agar tidak bergantung pada working directory
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "svc_pipeline.pkl"

# ── Load pipeline satu kali saat module pertama kali di-import ───────────────
# Tidak perlu load ulang setiap kali node dipanggil (lebih efisien)
_pipeline = joblib.load(MODEL_PATH)

# Pipeline sebelum classifier: prep -> scale -> fs
_feature_pipeline = _pipeline[:-1]

# Urutan input mentah harus sama dengan data training sebelum preprocessing.
FEATURE_COLUMNS = [
    "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges"
]


def preprocess_node(state: AgentState) -> AgentState:
    """
    Node kedua dalam graph. Tugasnya:
    1. Ambil customer_features dari state
    2. Ubah dict -> DataFrame dengan urutan kolom yang benar
    3. Jalankan transform fitur (prep + scale + fs)
    4. Simpan hasil array ke processed_features
    """

    # ── LANGKAH 1: Ambil data dari state ─────────────────────────────────────
    logger.info("[preprocess_node] START — memproses fitur customer")
    customer_features = state.get("customer_features")

    if not state.get("input_valid") or customer_features is None:
        logger.error("[preprocess_node] FAILED — customer_features is None")
        return {
            **state,
            "processed_features": None,
            "error_message": "customer_features is None, cannot preprocess"
        }

    # ── LANGKAH 2: Ubah dict → DataFrame (1 baris) ───────────────────────────
    df = pd.DataFrame([customer_features], columns=FEATURE_COLUMNS)

    # ── LANGKAH 3: Transform menggunakan feature pipeline yang sudah dilatih ──
    try:
        processed = _feature_pipeline.transform(df)  # output: numpy array shape (1, n_selected_features)
        logger.debug("[preprocess_node] transform OK — output shape: {}", processed.shape)
    except Exception as e:
        logger.error("[preprocess_node] FAILED — transform error: {}", e)
        return {
            **state,
            "processed_features": None,
            "error_message": f"Preprocessing failed: {str(e)}"
        }

    # ── LANGKAH 4: Simpan ke state ────────────────────────────────────────────
    logger.success("[preprocess_node] OK — fitur berhasil dipreprocess")
    return {
        **state,
        "processed_features": processed,
        "error_message": None
    }