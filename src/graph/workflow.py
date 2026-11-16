from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.graph.state import GraphState
from src.graph.nodes import understand_node, retrieve_node, extract_node, generate_node

# Initialize the StateGraph with our GraphState
workflow = StateGraph(GraphState)

# understand question -> retrieve chunks -> extract relevant facts -> write final answer
workflow.add_node("understand", understand_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("extract", extract_node)
workflow.add_node("generate", generate_node)

workflow.add_edge(START, "understand")
workflow.add_edge("understand", "retrieve")
workflow.add_edge("retrieve", "extract")
workflow.add_edge("extract", "generate")
workflow.add_edge("generate", END)

# Initialize in-memory checkpointer for conversation memory persistence
checkpointer = MemorySaver()

# Compile the workflow with the checkpointer enabled
app = workflow.compile(checkpointer=checkpointer)
