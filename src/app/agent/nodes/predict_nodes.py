import joblib
from pathlib import Path
from loguru import logger
from langgraph.graph import END
from ..schema import AgentState

# 1. Ubah path untuk menunjuk ke model pipeline SVC yang baru
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "svc_pipeline.pkl"
_pipeline = joblib.load(MODEL_PATH)

_classifier = _pipeline.named_steps["clf"]

def predict_node(state: AgentState) -> AgentState:
    """
    Node ini menjalankan prediksi churn menggunakan fitur yang sudah dipreprocess
    dari preprocess_node agar alur graph tetap modular.
    """
    logger.info("[predict_node] START — menjalankan prediksi churn")

    preprocessed_features = state.get("processed_features")

    if preprocessed_features is None:
        logger.warning("[predict_node] SKIP — processed_features is None, kemungkinan preprocess gagal")
        return {
            **state,
            'prediction': None,
            'churn_probability': None,
        }
    
    try:
        predict = _classifier.predict(preprocessed_features)[0]
        probs = _classifier.predict_proba(preprocessed_features)[0]

    except Exception as e:
        logger.error("[predict_node] FAILED — classifier error: {}", e)
        return {
            **state,
            'prediction': None,
            'churn_probability': None,
            'error_message': f"Prediction failed: {str(e)}"
        }

    label = "CHURN" if int(predict) == 1 else "NO CHURN"
    churn_prob = float(probs[1])
    logger.success("[predict_node] OK — hasil: {} (probability: {:.2%})", label, churn_prob)
    
    return {
        **state,
        "prediction": int(predict),
        "churn_probability": churn_prob,
        "error_message": None
    }

def route_by_prediction(state: AgentState) -> str:
    """Conditional edge function untuk routing berdasarkan hasil prediksi."""

    prediction = state.get("prediction")

    if prediction is None:
        logger.warning("[router] 'prediction' is None — graph berhenti (END)")
        return END

    if prediction == 1:
        logger.info("[router] routing ke → churn_prevention_agent")
        return "churn_prevention_agent"
    else:
        logger.info("[router] routing ke → retention_enhancement_agent")
        return "retention_enhancement_agent"
