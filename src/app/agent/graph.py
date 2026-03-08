# graph.py (preview)
from langgraph.graph import StateGraph, END
from src.app.agent.schema import AgentState
from src.app.agent.nodes.input_nodes import input_node
from src.app.agent.nodes.preprocess_nodes import preprocess_node 

def route_after_input(state: AgentState) -> str:
    # Jika valid → lanjut ke preprocess
    # Jika tidak valid → berhenti
    return "preprocess" if state["input_valid"] else END

graph = StateGraph(AgentState)
graph.add_node("input", input_node)  
graph.add_node("preprocess", preprocess_node)  
        # daftarkan node
graph.set_entry_point("input")               # mulai dari sini
graph.add_conditional_edges(                 # routing setelah input
    "input",
    route_after_input,
    {"preprocess": "preprocess", END: END}
)
graph.add_edge("preprocess", "predict") 