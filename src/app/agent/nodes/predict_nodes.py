import joblib
from pathlib import Path
from langgraph.graph import END
from ..schema import AgentState

MODEL_PATH = Path(__file__).resolve().parents[2]/ "models" / "xgb_undersampling_pipeline.pkl"
_pipeline = joblib.load(MODEL_PATH)
_classifier = _pipeline.named_steps["clf"]

def predict_node(state: AgentState) -> AgentState:

    preprocesed_features = state["processed_features"]

    if preprocesed_features is None:
        return {
            **state,
            'prediction': None,
            'churn_probability': None,
            'error_message': "preprocessed_features is  None, cannot predict"
        }
    
    try:
        predict = _classifier.predict(preprocesed_features)[0]
        probs = _classifier.predict_proba(preprocesed_features)[0]
    except Exception as e:
        return {
            **state,
            'prediction': None,
            'churn_probability': None,
            'error_message': f"Prediction failed: {str(e)}"
        }
    
    return {
        **state,
        "prediction": int(predict),
        "churn_probability": float(probs[1]),
        "error_message": None
    }

def route_by_prediction(state: AgentState) -> str:
    """Conditional edge fuction"""

    if state['prediction'] is None:
        return END

    if state['prediction'] == 1:
        return "churn_prevention_agent"
    else:
        return "retention_enhancement_agent"
    
