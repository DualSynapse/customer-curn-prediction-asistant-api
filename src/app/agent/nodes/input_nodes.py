# src/app/agent/nodes/input_nodes.py

import json
from loguru import logger
from src.app.agent.schema import AgentState, CustomerFeatures


# ── Daftar semua field wajib beserta tipe yang diharapkan ─────────────────────
# Disesuaikan dengan fitur yang digunakan di pipeline `train_svc.ipynb`
# 'gender' dan 'TotalCharges' dihapus karena tidak digunakan dalam model.
# Tipe data diubah menjadi string untuk mencerminkan input mentah sebelum preprocessing.
REQUIRED_FIELDS = {
    "SeniorCitizen": str,
    "tenure": int,
    "PhoneService": str,
    "MultipleLines": str,
    "InternetService": str,
    "OnlineSecurity": str,
    "OnlineBackup": str,
    "DeviceProtection": str,
    "TechSupport": str,
    "StreamingTV": str,
    "StreamingMovies": str,
    "Contract": str,
    "PaperlessBilling": str,
    "PaymentMethod": str,
    "MonthlyCharges": float,
    "FamilyStatus": str,
}


# ── Fungsi bantu: validasi nilai ──────────────────────────────────────────────
def _validate_values(data: dict) -> list[str]:
    """
    Cek apakah nilai-nilai dalam data masuk akal secara bisnis,
    sesuai dengan nilai mentah yang diharapkan oleh pipeline preprocessing.
    Kembalikan list error string (kosong jika semua valid).
    """
    errors = []

    # tenure tidak boleh negatif
    if data.get("tenure", 0) < 0:
        errors.append("tenure cannot be negative")

    # MonthlyCharges tidak boleh negatif
    if data.get("MonthlyCharges", 0) < 0:
        errors.append("MonthlyCharges cannot be negative")

    # Validasi untuk field string "Yes" atau "No"
    string_yes_no_fields = ["PhoneService", "PaperlessBilling"]
    for field in string_yes_no_fields:
        if field in data and data[field] not in ("Yes", "No"):
            errors.append(f"'{field}' must be 'Yes' or 'No', got: '{data[field]}'")

    # SeniorCitizen: harus string "1" atau "0"
    if "SeniorCitizen" in data and str(data["SeniorCitizen"]) not in ("0", "1"):
        errors.append(f"'SeniorCitizen' must be '0' or '1', got: '{data['SeniorCitizen']}'")

    # MultipleLines: "Yes", "No", atau "No phone service"
    if "MultipleLines" in data and data["MultipleLines"] not in ("Yes", "No", "No phone service"):
        errors.append(f"'MultipleLines' must be 'Yes', 'No', or 'No phone service', got: '{data['MultipleLines']}'")

    # InternetService: "DSL", "Fiber optic", atau "No"
    if "InternetService" in data and data["InternetService"] not in ("DSL", "Fiber optic", "No"):
        errors.append(f"'InternetService' must be 'DSL', 'Fiber optic', or 'No', got: '{data['InternetService']}'")

    # Layanan internet lainnya: "Yes", "No", atau "No internet service"
    internet_service_fields = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    for field in internet_service_fields:
        if field in data and data[field] not in ("Yes", "No", "No internet service"):
            errors.append(f"'{field}' must be 'Yes', 'No', or 'No internet service', got: '{data[field]}'")

    # Contract: "Month-to-month", "One year", atau "Two year"
    if "Contract" in data and data["Contract"] not in ("Month-to-month", "One year", "Two year"):
        errors.append(f"'Contract' must be 'Month-to-month', 'One year', or 'Two year', got: '{data['Contract']}'")

    # PaymentMethod: harus salah satu dari 4 pilihan yang valid
    valid_payment_methods = [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    if "PaymentMethod" in data and data["PaymentMethod"] not in valid_payment_methods:
        errors.append(f"Invalid 'PaymentMethod', got: '{data['PaymentMethod']}'")

    # FamilyStatus: gunakan kategori yang terlihat pada data training
    valid_family_statuses = ["Couple", "Family", "Single", "Single Parent"]
    if "FamilyStatus" in data and data["FamilyStatus"] not in valid_family_statuses:
        errors.append(f"Invalid 'FamilyStatus', got: '{data['FamilyStatus']}'")

    return errors


# ── Input Node Utama ──────────────────────────────────────────────────────────
def input_node(state: AgentState) -> AgentState:
    """
    Node pertama dalam graph. Tugasnya:
    1. Parse user_message dari JSON string menjadi dict.
    2. Cek apakah semua field wajib ada.
    3. Cek apakah tipe data tiap field benar.
    4. Cek apakah nilainya masuk akal (sesuai nilai mentah).
    5. Simpan hasilnya ke state untuk digunakan oleh node prediksi.
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
    type_errors = []
    for field, expected_type in REQUIRED_FIELDS.items():
        value = data[field]
        if expected_type == float and not isinstance(value, (int, float)):
            type_errors.append(f"'{field}' must be a number, got: {type(value).__name__}")
        elif expected_type == int and not isinstance(value, int):
            type_errors.append(f"'{field}' must be an integer, got: {type(value).__name__}")
        elif expected_type == str and not isinstance(value, str):
            # Khusus untuk SeniorCitizen, kita terima int 0/1 dan akan di-handle sebagai str
            if field == "SeniorCitizen" and isinstance(value, int) and value in (0, 1):
                data[field] = str(value) # Konversi int ke str
                continue
            type_errors.append(f"'{field}' must be a string, got: {type(value).__name__}")

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
        field: data[field] for field in REQUIRED_FIELDS
    }

    logger.success("[input_node] OK — input valid, semua field lengkap")
    return {
        **state,
        "customer_features": customer_features,
        "input_valid": True,
        "error_message": None
    }