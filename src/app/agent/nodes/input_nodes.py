# src/app/agent/nodes/input_nodes.py

import json
from loguru import logger
from src.app.agent.schema import AgentState, CustomerFeatures


# ── Daftar semua field wajib beserta tipe yang diharapkan ─────────────────────
REQUIRED_FIELDS = {
    "gender": int,
    "SeniorCitizen": int,
    "Partner": int,
    "Dependents": int,
    "tenure": int,
    "PhoneService": int,
    "MultipleLines": int,
    "InternetService": int,
    "OnlineSecurity": int,
    "OnlineBackup": int,
    "DeviceProtection": int,
    "TechSupport": int,
    "StreamingTV": int,
    "StreamingMovies": int,
    "Contract": int,
    "PaperlessBilling": int,
    "PaymentMethod": int,
    "MonthlyCharges": float,
    "TotalCharges": float,
}


# ── Fungsi bantu: validasi nilai ──────────────────────────────────────────────
def _validate_values(data: dict) -> list[str]:
    """
    Cek apakah nilai-nilai dalam data masuk akal secara bisnis.
    Kembalikan list error string (kosong jika semua valid).
    """
    errors = []

    # tenure tidak boleh negatif
    if data.get("tenure", 0) < 0:
        errors.append("tenure cannot be negative")

    # MonthlyCharges tidak boleh negatif
    if data.get("MonthlyCharges", 0) < 0:
        errors.append("MonthlyCharges cannot be negative")

    # TotalCharges tidak boleh negatif
    if data.get("TotalCharges", 0) < 0:
        errors.append("TotalCharges cannot be negative")

    # Field biner hanya boleh 0 atau 1
    binary_fields = [
        "gender", "SeniorCitizen", "Partner", "Dependents",
        "PhoneService", "MultipleLines", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "PaperlessBilling"
    ]
    for field in binary_fields:
        if field in data and data[field] not in (0, 1):
            errors.append(f"{field} must be 0 or 1, got: {data[field]}")

    # InternetService: hanya boleh 0, 1, atau 2
    if "InternetService" in data and data["InternetService"] not in (0, 1, 2):
        errors.append("InternetService must be 0, 1, or 2")

    # Contract: hanya boleh 0, 1, atau 2
    if "Contract" in data and data["Contract"] not in (0, 1, 2):
        errors.append("Contract must be 0, 1, or 2")

    # PaymentMethod: hanya boleh 0, 1, 2, atau 3
    if "PaymentMethod" in data and data["PaymentMethod"] not in (0, 1, 2, 3):
        errors.append("PaymentMethod must be 0, 1, 2, or 3")

    return errors


# ── Input Node Utama ──────────────────────────────────────────────────────────
def input_node(state: AgentState) -> AgentState:
    """
    Node pertama dalam graph. Tugasnya:
    1. Parse user_message dari JSON string menjadi dict
    2. Cek apakah semua field wajib ada
    3. Cek apakah tipe data tiap field benar
    4. Cek apakah nilainya masuk akal
    5. Simpan hasilnya ke state
    """

    raw_input = state["user_message"]
    logger.info("[input_node] START — memvalidasi input customer")

    # ── LANGKAH 1: Parse JSON ─────────────────────────────────────────────────
    try:
        data: dict = json.loads(raw_input)
        logger.debug("[input_node] JSON parsed OK")
    except json.JSONDecodeError as e:
        logger.error("[input_node] FAILED — invalid JSON: {}", e)
        return {
            **state,
            "customer_features": None,
            "input_valid": False,
            "error_message": f"Invalid JSON format: {str(e)}"
        }

    # ── LANGKAH 2: Cek field yang tidak ada (missing) ─────────────────────────
    missing_fields = [
        field for field in REQUIRED_FIELDS if field not in data
    ]
    if missing_fields:
        logger.warning("[input_node] FAILED — missing fields: {}", missing_fields)
        return {
            **state,
            "customer_features": None,
            "input_valid": False,
            "error_message": f"Missing required fields: {missing_fields}"
        }

    # ── LANGKAH 3: Cek tipe data tiap field ──────────────────────────────────
    # Contoh: tenure harus int, MonthlyCharges harus float/int
    type_errors = []
    for field, expected_type in REQUIRED_FIELDS.items():
        value = data[field]
        # float bisa menerima int (5 → 5.0), tapi int tidak bisa menerima string
        if expected_type == float:
            if not isinstance(value, (int, float)):
                type_errors.append(
                    f"{field} must be a number, got: {type(value).__name__}"
                )
        elif not isinstance(value, expected_type):
            type_errors.append(
                f"{field} must be {expected_type.__name__}, got: {type(value).__name__}"
            )

    if type_errors:
        logger.warning("[input_node] FAILED — type errors: {}", type_errors)
        return {
            **state,
            "customer_features": None,
            "input_valid": False,
            "error_message": f"Type errors: {type_errors}"
        }

    # ── LANGKAH 4: Validasi nilai (business rules) ────────────────────────────
    value_errors = _validate_values(data)
    if value_errors:
        logger.warning("[input_node] FAILED — value errors: {}", value_errors)
        return {
            **state,
            "customer_features": None,
            "input_valid": False,
            "error_message": f"Value errors: {value_errors}"
        }

    # ── LANGKAH 5: Semua valid — ekstrak dan simpan ke state ─────────────────
    # Ambil hanya field yang diperlukan (abaikan field extra yang tidak dikenal)
    customer_features: CustomerFeatures = {
        field: float(data[field]) if expected_type == float else int(data[field])
        for field, expected_type in REQUIRED_FIELDS.items()
    }

    logger.success("[input_node] OK — input valid, semua field lengkap")
    return {
        **state,
        "customer_features": customer_features,
        "input_valid": True,
        "error_message": None
    }