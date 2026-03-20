import joblib
from pathlib import Path
from loguru import logger
from langgraph.graph import END
from ..schema import AgentState

MODEL_PATH = Path(__file__).resolve().parents[2]/ "models" / "xgb_undersampling_pipeline.pkl"
_pipeline = joblib.load(MODEL_PATH)
_classifier = _pipeline.named_steps["clf"]

def predict_node(state: AgentState) -> AgentState:
    logger.info("[predict_node] START — menjalankan prediksi churn")

    preprocesed_features = state["processed_features"]

    if preprocesed_features is None:
        logger.warning("[predict_node] SKIP — processed_features is None, kemungkinan preprocess gagal")
        return {
            **state,
            'prediction': None,
            'churn_probability': None,
        }
    
    try:
        predict = _classifier.predict(preprocesed_features)[0]
        probs = _classifier.predict_proba(preprocesed_features)[0]
    except Exception as e:
        logger.error("[predict_node] FAILED — classifier error: {}", e)
        return {
            **state,
            'prediction': None,
            'churn_probability': None,
            'error_message': f"Prediction failed: {str(e)}"
        }

    label = "CHURN" if int(predict) == 1 else "NO CHURN"
    logger.success("[predict_node] OK — hasil: {} (probability: {:.2%})", label, float(probs[1]))
    return {
        **state,
        "prediction": int(predict),
        "churn_probability": float(probs[1]),
        "error_message": None
    }

def route_by_prediction(state: AgentState) -> str:
    """Conditional edge function"""

    if state['prediction'] is None:
        logger.warning("[router] prediction is None — graph berhenti (END)")
        return END

    if state['prediction'] == 1:
        logger.info("[router] routing ke → churn_prevention_agent")
        return "churn_prevention_agent"
    else:
        logger.info("[router] routing ke → retention_enhancement_agent")
        return "retention_enhancement_agent"
    
