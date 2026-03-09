import json
from src.app.agent.graph import app as agent_graph

def predict_churn(customer_data: dict) -> dict:
    initial_state = {
        'user_message': json.dumps(customer_data),
        'input_valid': False,
        'customer_features': None,
        'processed_features': None,
        'prediction': None,
        'churn_probability': None,
        'recommendation': None,
        'error_message': None,
        'final_response': None,
    }
    result = agent_graph.invoke(initial_state)

    # final_response bisa None jika graph berhenti lebih awal (input invalid / predict gagal)
    # sebelum response_node sempat dijalankan
    if result.get('final_response') is not None:
        return result['final_response']

    return {
        "churn_risk": None,
        "churn_probability": None,
        "recommendation": None,
        "error": result.get('error_message', 'Graph stopped early without a response'),
    }

