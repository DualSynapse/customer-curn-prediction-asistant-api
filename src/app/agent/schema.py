from typing import TypedDict, Optional
import numpy as np
from numpy.typing import NDArray
# struktur data customer
class CustomerFeatures(TypedDict):
    SeniorCitizen: str       # "0" atau "1"
    Partner: str             # "Yes" / "No"
    Dependents: str          # "Yes" / "No"
    tenure: int              # jumlah bulan berlangganan
    PhoneService: str        # "Yes" / "No"
    MultipleLines: str       # "Yes" / "No" / "No phone service"
    InternetService: str     # "DSL" / "Fiber optic" / "No"
    OnlineSecurity: str      # "Yes" / "No" / "No internet service"
    OnlineBackup: str        # "Yes" / "No" / "No internet service"
    DeviceProtection: str    # "Yes" / "No" / "No internet service"
    TechSupport: str         # "Yes" / "No" / "No internet service"
    StreamingTV: str         # "Yes" / "No" / "No internet service"
    StreamingMovies: str     # "Yes" / "No" / "No internet service"
    Contract: str            # "Month-to-month" / "One year" / "Two year"
    PaperlessBilling: str    # "Yes" / "No"
    PaymentMethod: str       # metode pembayaran kategorikal
    MonthlyCharges: float    # tagihan bulanan

# ── 2. State global yang dibawa seluruh graph ──────────────────────────────────
# Setiap node membaca dan menulis ke sini
class AgentState(TypedDict):

    # --- Diisi oleh user/API (sebelum graph jalan) ---
    user_message: str                           # raw input dari user (JSON string)

    # --- Diisi oleh input_node ---
    customer_features: Optional[CustomerFeatures]  # data customer yang sudah terstruktur
    input_valid: bool                           # True jika input valid, False jika tidak
    error_message: Optional[str]               # pesan error jika input tidak valid

    # --- Diisi oleh preprocess_node (nanti) ---
    processed_features: Optional[NDArray]

    # --- Diisi oleh predict_node (nanti) ---
    churn_probability: Optional[float]
    prediction: Optional[int]                  # 0 = tidak churn, 1 = churn

    # --- Diisi oleh retention/prevention node (nanti) ---
    recommendation: Optional[str]

    # --- Diisi oleh response_node ---
    final_response: Optional[dict]
