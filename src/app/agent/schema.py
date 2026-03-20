from typing import TypedDict, Optional
import numpy as np
from numpy.typing import NDArray
# struktur data customer
class CustomerFeatures(TypedDict):
    gender: int             # 0 = Female, 1 = Male
    SeniorCitizen: int      # 0 = No, 1 = Yes
    Partner: int            # 0 = No, 1 = Yes
    Dependents: int         # 0 = No, 1 = Yes
    tenure: int             # jumlah bulan berlangganan
    PhoneService: int       # 0 = No, 1 = Yes
    MultipleLines: int      # 0 = No, 1 = Yes
    InternetService: int    # 0 = No, 1 = DSL, 2 = Fiber optic
    OnlineSecurity: int     # 0 = No, 1 = Yes
    OnlineBackup: int       # 0 = No, 1 = Yes
    DeviceProtection: int   # 0 = No, 1 = Yes
    TechSupport: int        # 0 = No, 1 = Yes
    StreamingTV: int        # 0 = No, 1 = Yes
    StreamingMovies: int    # 0 = No, 1 = Yes
    Contract: int           # 0 = Month-to-month, 1 = One year, 2 = Two year
    PaperlessBilling: int   # 0 = No, 1 = Yes
    PaymentMethod: int      # 0~3 (berbagai metode pembayaran)
    MonthlyCharges: float   # tagihan bulanan
    TotalCharges: float     # total tagihan

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
