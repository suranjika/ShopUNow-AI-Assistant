"""
human_node.py
-------------
Terminal node for escalated queries.
In production, this would trigger a CRM ticket or support queue notification.
"""

from graphs.state import ChatState


def human_node(state: ChatState) -> ChatState:
    """
    Handles escalated queries.
    Sets a final escalation response message.
    """
    state["response"] = (
        "Your query has been escalated to a human support agent. "
        "Please provide your name, phone, and email via the API request body "
        "and our team will reach out to you shortly. 🙋"
    )
    return state
