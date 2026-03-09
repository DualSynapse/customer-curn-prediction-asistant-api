from langgraph.graph import StateGraph, END
from src.app.agent.schema import AgentState
from src.app.agent.nodes.input_nodes import input_node
from src.app.agent.nodes.preprocess_nodes import preprocess_node
from src.app.agent.nodes.predict_nodes import predict_node, route_by_prediction
from src.app.agent.nodes.prevention_nodes import prevention_node
from src.app.agent.nodes.retention_nodes import retention_node
from src.app.agent.nodes.response_nodes import response_node


def route_after_input(state: AgentState) -> str:
    return "preprocess_node" if state["input_valid"] else END


def build_churn_agent_graph():
    graph = StateGraph(AgentState)

    graph.add_node("input_node", input_node)
    graph.add_node("preprocess_node", preprocess_node)
    graph.add_node("predict_node", predict_node)
    graph.add_node("churn_prevention_agent", prevention_node)
    graph.add_node("retention_enhancement_agent", retention_node)
    graph.add_node("response_formater", response_node)

    graph.set_entry_point("input_node")

    graph.add_conditional_edges(
        "input_node",
        route_after_input,
        {"preprocess_node": "preprocess_node", END: END}
    )

    graph.add_conditional_edges(
        "preprocess_node",
        lambda state: "predict_node" if state["processed_features"] is not None else END,
        {"predict_node": "predict_node", END: END}
    )

    graph.add_conditional_edges(
        "predict_node",
        route_by_prediction,
        {
            "churn_prevention_agent": "churn_prevention_agent",
            "retention_enhancement_agent": "retention_enhancement_agent",
            END: END,
        }
    )

    graph.add_edge("churn_prevention_agent", "response_formater")
    graph.add_edge("retention_enhancement_agent", "response_formater")
    graph.add_edge("response_formater", END)

    return graph.compile()


app = build_churn_agent_graph()
