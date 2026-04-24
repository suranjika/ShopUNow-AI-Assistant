"""
main.py
-------
FastAPI application for ShopUNow Agentic AI Assistant.

Endpoints:
  GET  /health  — Health check
  POST /ask     — Submit a query to the agent

Run with:
  uvicorn api.main:api --reload --host 0.0.0.0 --port 8000
"""

import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv

from graphs.state import ChatState
from graphs.workflow import app as agent_app
from rag.knowledge_base import DEPARTMENTS

# Allow nested async only in notebook/interactive environments — not in production server
if "ipykernel" in sys.modules:
    import nest_asyncio
    nest_asyncio.apply()

# Load environment variables from .env
load_dotenv()

# ─── FastAPI app ──────────────────────────────────────────────────────────────

api = FastAPI(
    title="ShopUNow Agentic AI Assistant",
    description=(
        "Enterprise-grade multi-agent AI assistant for HR, Finance, Billing, "
        "and Shipping queries. Built with LangGraph, LangChain, Gemini, and FAISS."
    ),
    version="1.0.0",
)


# ─── Request / Response Schemas ───────────────────────────────────────────────

class AskRequest(BaseModel):
    query:  str
    name:   str | None = None
    phone:  str | None = None
    email:  EmailStr | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "How do I apply for annual leave?",
                },
                {
                    "query": "My delivery is late and I'm very frustrated!",
                    "name": "John Doe",
                    "phone": "+62 812 3456 7890",
                    "email": "john@example.com",
                },
            ]
        }
    }


class AskResponse(BaseModel):
    response:        str
    sentiment:       str
    department:      str
    escalated:       bool
    need_details:    bool
    required_fields: list[str] = []
    echo_details:    dict | None = None


# ─── Endpoints ────────────────────────────────────────────────────────────────

@api.get("/health", tags=["System"])
def health():
    """Health check — returns OK and list of supported departments."""
    return {"ok": True, "departments": DEPARTMENTS}


@api.post("/ask", response_model=AskResponse, tags=["Assistant"])
def ask(req: AskRequest):
    """
    Submit a query to the ShopUNow Agentic AI Assistant.

    Behaviour:
    - Standard queries are answered via the RAG pipeline (FAISS + Gemini).
    - Negative sentiment or unrecognised department queries are escalated to a human agent.
    - Escalated queries require name, phone, and email to complete the handoff.
    """
    # Build initial state for the LangGraph workflow
    initial_state = ChatState(
        message=req.query,
        sentiment="",
        department="",
        retrieved_docs=[],
        response="",
        escalate=False,
    )

    # Run through LangGraph workflow — wrapped in try/except for clean error responses
    try:
        result = agent_app.invoke(initial_state)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent workflow failed: {str(e)}"
        )

    # Extract results directly from TypedDict keys
    sentiment  = result["sentiment"]
    department = result["department"]
    escalated  = result["escalate"]

    # ── Case 1: Not escalated — return RAG answer ──────────────────────────────
    if not escalated:
        return AskResponse(
            response=result["response"],
            sentiment=sentiment,
            department=department,
            escalated=False,
            need_details=False,
            required_fields=[],
            echo_details=None,
        )

    # ── Case 2: Escalated — check for required contact details ─────────────────
    missing = []
    if not req.name:  missing.append("name")
    if not req.phone: missing.append("phone")
    if not req.email: missing.append("email")

    if missing:
        return AskResponse(
            response=(
                "Your issue requires human attention. "
                "Please provide the following details: " + ", ".join(missing) + "."
            ),
            sentiment=sentiment,
            department=department,
            escalated=True,
            need_details=True,
            required_fields=missing,
            echo_details=None,
        )

    # ── Case 3: Escalated — all details provided, confirm handoff ──────────────
    confirmation = (
        f"Thank you, {req.name}! Your escalation has been received.\n"
        f"📞 Phone: {req.phone}\n"
        f"📧 Email: {req.email}\n"
        "A human support agent will reach out to you shortly."
    )

    return AskResponse(
        response=confirmation,
        sentiment=sentiment,
        department=department,
        escalated=True,
        need_details=False,
        required_fields=[],
        echo_details={"name": req.name, "phone": req.phone, "email": str(req.email)},
    )
