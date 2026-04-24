"""
router_node.py
--------------
Classifies the user query into a department:
HR | Finance | Billing | Shipping | Unknown
Uses keyword matching for speed and reliability.
"""

from graphs.state import ChatState


def router_node(state: ChatState) -> ChatState:
    """
    Classify query department from keywords.
    Routes to: HR, Finance, Billing, Shipping, or Unknown.
    """
    text = state["message"].lower()

    if any(w in text for w in ["leave", "pto", "holiday", "vacation", "performance", "handbook",
                                "harassment", "parental", "remote", "payroll", "benefits", "training"]):
        state["department"] = "HR"

    elif any(w in text for w in ["expense", "reimbursement", "invoice", "fiscal", "per diem",
                                  "cost center", "vendor", "capital", "travel card", "po ", "purchase order"]):
        state["department"] = "Finance"

    elif any(w in text for w in ["card declined", "billing", "bill", "payment", "promo code",
                                  "refund", "bnpl", "installment", "vat", "charge", "download invoice"]):
        state["department"] = "Billing"

    elif any(w in text for w in ["shipping", "delivery", "track", "package", "order", "ship",
                                  "return", "fragile", "expedited", "po box"]):
        state["department"] = "Shipping"

    else:
        state["department"] = "Unknown"

    return state
