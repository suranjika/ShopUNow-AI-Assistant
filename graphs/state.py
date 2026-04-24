"""
state.py
--------
Shared ChatState TypedDict used across all LangGraph nodes.
Every node reads from and writes to this state object.
"""

from typing import TypedDict, List
from langchain_core.documents import Document


class ChatState(TypedDict):
    message:        str             # User's input query
    sentiment:      str             # positive | neutral | negative
    department:     str             # HR | Finance | Billing | Shipping | Unknown
    retrieved_docs: List[Document]  # Documents retrieved from FAISS
    response:       str             # Final assistant response
    escalate:       bool            # Whether query should be routed to human agent
