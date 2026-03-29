# src/app/agent/nodes/preprocess_nodes.py

import joblib
import pandas as pd
from pathlib import Path
from loguru import logger
from src.app.agent.schema import AgentState
from src.app.models.svc_transformers import register_legacy_pickle_functions

# ── Path ke file model pipeline ──────────────────────────────────────────────
# Path dihitung dari lokasi file ini agar tidak bergantung pada working directory
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "svc_pipeline.pkl"

# ── Load pipeline satu kali saat module pertama kali di-import ───────────────
# Tidak perlu load ulang setiap kali node dipanggil (lebih efisien)
register_legacy_pickle_functions()
_pipeline = joblib.load(MODEL_PATH)

# Pipeline sebelum classifier: prep -> scale -> fs
_feature_pipeline = _pipeline[:-1]

# Urutan input mentah harus sama dengan data training sebelum preprocessing.
FEATURE_COLUMNS = [
    "SeniorCitizen", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges",
    "FamilyStatus"
]


def _normalize_inference_input(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same value normalization used before fitting the SVC pipeline."""
    normalized = df.copy()

    # Training replaced these service-specific values with plain "No" before fitting.
    replace_no_service_cols = [
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    for col in replace_no_service_cols:
        if col in normalized.columns:
            normalized[col] = normalized[col].replace(
                {
                    "No internet service": "No",
                    "No phone service": "No",
                }
            )

    # Keep categorical string formatting consistent with training data.
    for col in normalized.select_dtypes(include="object").columns:
        normalized[col] = normalized[col].astype(str).str.strip()

    return normalized


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
    df = _normalize_inference_input(df)

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