from loguru import logger
from src.app.agent.schema import AgentState


def response_node(state: AgentState) -> AgentState:
    """
    Node terakhir. Merakit final_response dict yang siap di-return FastAPI.
    """
    logger.info("[response_node] START — merakit final response")
    prediction = state.get("prediction")

    final_response = {
        "churn_risk": bool(prediction) if prediction is not None else None,
        "churn_probability": round(state.get("churn_probability") or 0.0, 4),
        "recommendation": state.get("recommendation"),
        "error": state.get("error_message"),
    }

    logger.success(
        "[response_node] DONE — churn_risk={}, probability={}",
        final_response["churn_risk"],
        final_response["churn_probability"]
    )
    return {
        **state,
        "final_response": final_response,
    }