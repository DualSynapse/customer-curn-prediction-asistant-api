from loguru import logger
from src.app.models.model import model
from src.app.agent.schema import AgentState
from src.config.tracing import get_tracer
from src.app.prompt.template import PREVENTION_SYSTEM_PROMPT
from langchain_core.messages import SystemMessage, HumanMessage

def _decode_features(features: dict) -> str:
    """Convert raw customer input features into a human-readable profile."""
    def _value(name: str, default: str = "N/A") -> str:
        value = features.get(name, default)
        return str(value)

    lines = [
        f"- Senior Citizen: {_value('SeniorCitizen')}",
        f"- Tenure: {_value('tenure')} months",
        f"- Phone Service: {_value('PhoneService')}",
        f"- Multiple Lines: {_value('MultipleLines')}",
        f"- Internet Service: {_value('InternetService')}",
        f"- Online Security: {_value('OnlineSecurity')}",
        f"- Online Backup: {_value('OnlineBackup')}",
        f"- Device Protection: {_value('DeviceProtection')}",
        f"- Tech Support: {_value('TechSupport')}",
        f"- Streaming TV: {_value('StreamingTV')}",
        f"- Streaming Movies: {_value('StreamingMovies')}",
        f"- Contract Type: {_value('Contract')}",
        f"- Paperless Billing: {_value('PaperlessBilling')}",
        f"- Payment Method: {_value('PaymentMethod')}",
        f"- Family Status: {_value('FamilyStatus')}",
        f"- Monthly Charges: ${float(features.get('MonthlyCharges', 0.0)):.2f}",
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
    logger.info("[prevention_node] START — membuat rekomendasi pencegahan churn (probability: {:.2%})", probability)

    customer_description = _decode_features(features)
    human_content = (
        f"Profil Customer:\n"
        f"{customer_description}\n\n"
        f"Hasil Prediksi: CHURN RISK = TRUE\n"
        f"Churn Probability: {probability:.1%}\n\n"
        "Analisis profil customer di atas dan berikan strategi pencegahan churn yang spesifik dan dapat dieksekusi."
    )

    messages = [
        SystemMessage(content=PREVENTION_SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]

    tracer = get_tracer()
    callbacks = [tracer] if tracer is not None else []

    logger.debug("[prevention_node] memanggil LLM...")
    try:
        response = model.invoke(messages, config={'callbacks': callbacks})
        recommendation = response.content
        logger.success("[prevention_node] OK — rekomendasi berhasil dibuat ({} chars)", len(recommendation))
    except Exception as e:
        logger.error("[prevention_node] FAILED — LLM error: {}", e)
        recommendation = f"Failed to generate recommendation: {str(e)}"

    return {
        **state,
        "recommendation": recommendation,
    }


