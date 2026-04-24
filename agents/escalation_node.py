"""
escalation_node.py
------------------
Decides whether to escalate to a human agent.
Escalates if: sentiment == "negative" OR department == "Unknown"
"""

from graphs.state import ChatState


def escalation_node(state: ChatState) -> ChatState:
    """
    Set escalate=True if the query needs human attention.
    Conditions:
    - Negative sentiment detected
    - Department could not be classified (Unknown)
    """
    if state["sentiment"] == "negative" or state["department"] == "Unknown":
        state["escalate"] = True
        state["response"] = (
            "Your query has been escalated to a human agent. "
            "You'll be contacted shortly. 🙋"
        )
    else:
        state["escalate"] = False
    return state
