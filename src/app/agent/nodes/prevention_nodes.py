from src.app.models.model import model
from src.app.agent.schema import AgentState
from src.app.prompt.template import PREVENTION_SYSTEM_PROMPT
from langchain_core.messages import SystemMessage, HumanMessage  

_GENDER      = {0: "Female", 1: "Male"}
_YES_NO      = {0: "No", 1: "Yes"}
_INTERNET    = {0: "No internet service", 1: "DSL", 2: "Fiber optic"}
_CONTRACT    = {0: "Month-to-month", 1: "One year", 2: "Two year"}
_PAYMENT     = {
    0: "Electronic check",
    1: "Mailed check",
    2: "Bank transfer (automatic)",
    3: "Credit card (automatic)"
}

def _decode_features(features: dict) -> str:
    """Ubah dict encoded (int/float) → paragraf teks deskriptif behavior customer."""
    lines = [
        f"- Gender: {_GENDER.get(features['gender'], features['gender'])}",
        f"- Senior Citizen: {_YES_NO.get(features['SeniorCitizen'])}",
        f"- Has Partner: {_YES_NO.get(features['Partner'])}",
        f"- Has Dependents: {_YES_NO.get(features['Dependents'])}",
        f"- Tenure: {features['tenure']} months",
        f"- Phone Service: {_YES_NO.get(features['PhoneService'])}",
        f"- Multiple Lines: {_YES_NO.get(features['MultipleLines'])}",
        f"- Internet Service: {_INTERNET.get(features['InternetService'])}",
        f"- Online Security: {_YES_NO.get(features['OnlineSecurity'])}",
        f"- Online Backup: {_YES_NO.get(features['OnlineBackup'])}",
        f"- Device Protection: {_YES_NO.get(features['DeviceProtection'])}",
        f"- Tech Support: {_YES_NO.get(features['TechSupport'])}",
        f"- Streaming TV: {_YES_NO.get(features['StreamingTV'])}",
        f"- Streaming Movies: {_YES_NO.get(features['StreamingMovies'])}",
        f"- Contract Type: {_CONTRACT.get(features['Contract'])}",
        f"- Paperless Billing: {_YES_NO.get(features['PaperlessBilling'])}",
        f"- Payment Method: {_PAYMENT.get(features['PaymentMethod'])}",
        f"- Monthly Charges: ${features['MonthlyCharges']:.2f}",
        f"- Total Charges: ${features['TotalCharges']:.2f}",
    ]
    return "\n".join(lines)


def prevention_node(state: AgentState) -> AgentState:
    """
    Churn prevention node. Alur:
    1. Decode customer_features → teks deskriptif
    2. Bungkus sebagai HumanMessage + SystemMessage
    3. Invoke LLM → dapat rekomendasi pencegahan churn
    4. Simpan ke state['recommendation']
    """
    features = state["customer_features"]
    probability = state.get("churn_probability") or 0.0

    # ── Bangun HumanMessage dari profil customer ──────────────────────────────
    customer_description = _decode_features(features)
    human_content = (
        f"Customer Profile (Churn Probability: {probability:.1%}):\n"
        f"{customer_description}\n\n"
        "Analisis profil customer di atas dan berikan strategi pencegahan churn yang spesifik dan dapat dieksekusi."
    )

    messages = [
        SystemMessage(content=PREVENTION_SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]

    # ── Invoke LLM ────────────────────────────────────────────────────────────
    try:
        response = model.invoke(messages)
        recommendation = response.content
    except Exception as e:
        recommendation = f"Failed to generate recommendation: {str(e)}"

    return {
        **state,
        "recommendation": recommendation,
    }


