"""
workflow.py
-----------
Defines and compiles the LangGraph StateGraph for ShopUNow Assistant.

Flow:
  [router] → [sentiment] → [escalation] → [human]  → END
                                        ↘ [rag]    → END
"""

from langgraph.graph import StateGraph, END
from graphs.state import ChatState
from agents.router_node import router_node
from agents.sentiment_node import sentiment_node
from agents.escalation_node import escalation_node
from agents.human_node import human_node
from agents.rag_node import build_rag_node
from rag.vectorstore import build_vectorstore


def build_graph():
    """
    Initialises the vector store, wires up all agent nodes,
    and returns a compiled LangGraph application.
    """
    # Build FAISS retriever once at startup
    _, retriever = build_vectorstore(k=2)
    rag_node = build_rag_node(retriever)

    # Initialise graph
    graph = StateGraph(ChatState)

    # Register nodes
    graph.add_node("router",     router_node)
    graph.add_node("sentiment",  sentiment_node)
    graph.add_node("escalation", escalation_node)
    graph.add_node("rag",        rag_node)
    graph.add_node("human",      human_node)

    # Define edges
    graph.set_entry_point("router")
    graph.add_edge("router",     "sentiment")
    graph.add_edge("sentiment",  "escalation")

    # Conditional routing from escalation node
    graph.add_conditional_edges(
        "escalation",
        lambda state: "human" if state["escalate"] else "rag",
        {"human": "human", "rag": "rag"},
    )

    graph.add_edge("human", END)
    graph.add_edge("rag",   END)

    return graph.compile()


# Singleton — compiled once on import
app = build_graph()
