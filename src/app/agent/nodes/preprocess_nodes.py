# src/app/agent/nodes/preprocess_nodes.py

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from src.app.agent.schema import AgentState

# ── Path ke file model pipeline ──────────────────────────────────────────────
# Path dihitung dari lokasi file ini agar tidak bergantung pada working directory
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "xgb_undersampling_pipeline.pkl"
# ── Load pipeline satu kali saat module pertama kali di-import ───────────────
# Tidak perlu load ulang setiap kali node dipanggil (lebih efisien)
_pipeline = joblib.load(MODEL_PATH)
preprocessor = _pipeline.named_steps["prep"]


# ── Urutan kolom HARUS sama persis dengan saat training ──────────────────────
FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]


def preprocess_node(state: AgentState) -> AgentState:
    """
    Node kedua dalam graph. Tugasnya:
    1. Ambil customer_features dari state
    2. Ubah dict → DataFrame dengan urutan kolom yang benar
    3. Jalankan preprocessor (StandardScaler + OneHotEncoder)
    4. Simpan hasil array ke processed_features
    """

    # ── LANGKAH 1: Ambil data dari state ─────────────────────────────────────
    customer_features = state["customer_features"]

    # Guard: seharusnya tidak sampai sini kalau input_valid = False
    # tapi sebagai safety check
    if customer_features is None:
        return {
            **state,
            "processed_features": None,
            "error_message": "customer_features is None, cannot preprocess"
        }

    # ── LANGKAH 2: Ubah dict → DataFrame (1 baris) ───────────────────────────
    # DataFrame dibutuhkan karena preprocessor (ColumnTransformer) dilatih
    # dengan DataFrame dan mengenali nama kolom
    df = pd.DataFrame([customer_features], columns=FEATURE_COLUMNS)

    # ── LANGKAH 3: Transform menggunakan preprocessor yang sudah dilatih ─────
    # preprocessor sudah di-fit saat training, kita hanya .transform() saja
    # (BUKAN .fit_transform() — itu hanya untuk training)
    try:
        processed = preprocessor.transform(df)  # output: numpy array shape (1, n_features)
    except Exception as e:
        return {
            **state,
            "processed_features": None,
            "error_message": f"Preprocessing failed: {str(e)}"
        }

    # ── LANGKAH 4: Simpan ke state ────────────────────────────────────────────
    return {
        **state,
        "processed_features": processed,
        "error_message": None
    }