"""
app.py
------
Streamlit frontend for ShopUNow Agentic AI Assistant.
Designed for deployment on Hugging Face Spaces.

Run locally:
  streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv
from typing import List, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END

# ── Load environment ───────────────────────────────────────────────────────────
load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ShopUNow AI Assistant",
    page_icon="🛍️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

* { font-family: 'DM Sans', sans-serif; }

/* Hide Streamlit branding */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* Main background */
.stApp { background: #0D1117; }

/* Header */
.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #F0F6FF;
    letter-spacing: -0.5px;
    margin-bottom: 0;
}
.hero-sub {
    font-size: 0.95rem;
    color: #8B9AB0;
    margin-top: 4px;
    margin-bottom: 24px;
}

/* Chat messages */
.msg-user {
    background: #1C2333;
    border: 1px solid #2A3547;
    border-radius: 12px 12px 4px 12px;
    padding: 14px 18px;
    margin: 8px 0;
    color: #E2E8F0;
    font-size: 0.95rem;
    text-align: right;
}
.msg-bot {
    background: #112240;
    border: 1px solid #1E3A5F;
    border-radius: 12px 12px 12px 4px;
    padding: 14px 18px;
    margin: 8px 0;
    color: #E2E8F0;
    font-size: 0.95rem;
}
.msg-escalated {
    background: #2D1B1B;
    border: 1px solid #5C2626;
    border-radius: 12px;
    padding: 14px 18px;
    margin: 8px 0;
    color: #FCA5A5;
    font-size: 0.95rem;
}

/* Department badge */
.dept-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-right: 6px;
    font-family: 'DM Mono', monospace;
}
.badge-HR       { background: #0F3460; color: #60A5FA; border: 1px solid #1E40AF; }
.badge-Finance  { background: #0F3D2E; color: #34D399; border: 1px solid #065F46; }
.badge-Billing  { background: #3D2C0F; color: #FBBF24; border: 1px solid #92400E; }
.badge-Shipping { background: #2D1B69; color: #A78BFA; border: 1px solid #4C1D95; }
.badge-Unknown  { background: #1F2937; color: #9CA3AF; border: 1px solid #374151; }

/* Sentiment indicator */
.sentiment-positive { color: #34D399; font-size: 0.78rem; }
.sentiment-neutral  { color: #9CA3AF; font-size: 0.78rem; }
.sentiment-negative { color: #F87171; font-size: 0.78rem; }

/* Input box */
.stTextInput input {
    background: #1C2333 !important;
    border: 1px solid #2A3547 !important;
    border-radius: 10px !important;
    color: #E2E8F0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 12px 16px !important;
}
.stTextInput input:focus {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.2) !important;
}

/* Buttons */
.stButton > button {
    background: #1D4ED8 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 10px 24px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #2563EB !important;
    transform: translateY(-1px) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0D1117 !important;
    border-right: 1px solid #1E2D3D !important;
}
.sidebar-title {
    font-size: 1rem;
    font-weight: 600;
    color: #F0F6FF;
    margin-bottom: 4px;
}
.sidebar-sub {
    font-size: 0.78rem;
    color: #8B9AB0;
    margin-bottom: 16px;
}

/* Escalation form */
.escalation-box {
    background: #1A0F0F;
    border: 1px solid #5C2626;
    border-radius: 12px;
    padding: 20px;
    margin: 12px 0;
}
.escalation-title {
    color: #FCA5A5;
    font-weight: 600;
    font-size: 0.95rem;
    margin-bottom: 8px;
}

/* Divider */
.custom-divider {
    border: none;
    border-top: 1px solid #1E2D3D;
    margin: 16px 0;
}

/* Stats pills */
.stat-pill {
    background: #1C2333;
    border: 1px solid #2A3547;
    border-radius: 8px;
    padding: 10px 14px;
    text-align: center;
    margin-bottom: 8px;
}
.stat-num { font-size: 1.4rem; font-weight: 700; color: #60A5FA; }
.stat-label { font-size: 0.72rem; color: #8B9AB0; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)


# ── Knowledge Base ─────────────────────────────────────────────────────────────
DATASETS = {
    "HR": [
        {"q": "How do I apply for paid time off (PTO)?", "a": "Submit a PTO request in Workday > Time Off. Manager approval required. Submit at least 3 business days in advance."},
        {"q": "What are ShopUNow's core working hours?", "a": "Core hours are 10:00–16:00 local time. Flexible start/end as agreed with your manager."},
        {"q": "Does ShopUNow offer parental leave?", "a": "Yes. 16 weeks paid primary caregiver leave and 6 weeks paid secondary caregiver leave."},
        {"q": "How do I update my legal name?", "a": "Open an HR ticket with legal documentation. HR will update payroll and directory within 5 business days."},
        {"q": "How can I access the employee handbook?", "a": "The handbook is in the HR Portal > Documents. Latest version is always pinned."},
        {"q": "How do performance reviews work?", "a": "Biannual reviews via Workday: self-review, peer/manager feedback, calibration, final rating."},
        {"q": "Can I work remotely?", "a": "Remote/hybrid depends on role and manager approval. See Remote Work Policy in HR Portal."},
        {"q": "Where do I report harassment?", "a": "Use the confidential Ethics & Compliance form or contact HRBP immediately."},
        {"q": "How do I enroll in benefits?", "a": "Enroll during onboarding or open enrollment via Benefits Center in Workday."},
        {"q": "What training is mandatory?", "a": "Security, Code of Conduct, and Anti-harassment trainings annually via LMS."},
    ],
    "Finance": [
        {"q": "What is the expense reimbursement policy?", "a": "Submit expenses within 30 days via Concur. Receipts required for items > $25. Travel must be pre-approved."},
        {"q": "How long do reimbursements take?", "a": "Approved claims are paid in the next weekly AP run (typically 5–7 business days)."},
        {"q": "What's the fiscal year?", "a": "ShopUNow fiscal year runs Jan 1 – Dec 31."},
        {"q": "How are per diems handled?", "a": "Per diems follow government/region tables; claim via Concur with travel dates and destination."},
        {"q": "How do vendor invoices get paid?", "a": "Vendors email invoices to ap@shopunow.com with PO. Net-30 unless negotiated."},
        {"q": "Who approves capital expenses?", "a": "Department head and Finance Controller must approve capex > $5,000."},
        {"q": "How do I request a PO?", "a": "Submit a PR in ProcureNow with vendor, quote, and cost center. PO created after approvals."},
        {"q": "Who to contact for payroll issues?", "a": "Open a Payroll ticket in the Finance Portal. Response in 1–2 business days."},
    ],
    "Billing": [
        {"q": "Why was my card declined?", "a": "Common reasons include insufficient funds or bank security checks. Try another card or contact your bank."},
        {"q": "How do I download my invoice?", "a": "Go to Account > Orders, select the order, then click Download Invoice."},
        {"q": "How do I apply a promo code?", "a": "Enter the code at checkout in the Promo field before payment."},
        {"q": "Refund timeline?", "a": "Once approved, refunds post to your bank in 5–10 business days."},
        {"q": "VAT invoice available?", "a": "Yes. Add your tax ID at checkout; invoice will include VAT details."},
        {"q": "How to update saved cards?", "a": "Account > Payment Methods to add/remove cards securely."},
        {"q": "My promo code isn't working.", "a": "Check expiration, minimum spend, or category exclusions; contact support if issues persist."},
    ],
    "Shipping": [
        {"q": "When will my order ship?", "a": "Orders ship within 1–2 business days. You'll receive a tracking email once dispatched."},
        {"q": "How do I track my package?", "a": "Use the tracking link in your email or go to Account > Orders and click Track."},
        {"q": "Do you offer expedited shipping?", "a": "Yes: Standard, Expedited (2-day), and Priority (next business day) at checkout."},
        {"q": "What if my package is late?", "a": "If tracking hasn't updated for 3 business days, contact support to open a carrier trace."},
        {"q": "Can I change the address after ordering?", "a": "Edits are possible within 30 minutes from Order Details > Edit Shipping Address."},
        {"q": "What is the return window?", "a": "30 days from delivery for most items. Start a return from Account > Returns."},
        {"q": "What are free-shipping thresholds?", "a": "Free Standard shipping on domestic orders over $50 (after discounts)."},
    ],
}

# ── LangGraph State ────────────────────────────────────────────────────────────
class ChatState(TypedDict):
    message:        str
    sentiment:      str
    department:     str
    retrieved_docs: List[Document]
    response:       str
    escalate:       bool

# ── Build Agent (cached) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Building AI agent — this takes ~30 seconds on first load...")
def build_agent():
    if not GOOGLE_API_KEY:
        return None, None

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    router_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    answer_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    # Build FAISS vector store
    docs = []
    for dept, entries in DATASETS.items():
        for entry in entries:
            docs.append(Document(
                page_content=entry["q"] + " " + entry["a"],
                metadata={"department": dept}
            ))
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    split_docs = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # Define nodes
    def router_node(state):
        text = state["message"].lower()
        if any(w in text for w in ["leave","pto","holiday","vacation","performance","handbook","harassment","parental","remote","payroll","benefits","training"]):
            state["department"] = "HR"
        elif any(w in text for w in ["expense","reimbursement","fiscal","per diem","cost center","vendor","capital","travel card","purchase order"]):
            state["department"] = "Finance"
        elif any(w in text for w in ["card declined","billing","bill","payment","promo code","refund","bnpl","installment","vat","charge","download invoice"]):
            state["department"] = "Billing"
        elif any(w in text for w in ["shipping","delivery","track","package","order","ship","return","expedited"]):
            state["department"] = "Shipping"
        else:
            state["department"] = "Unknown"
        return state

    def sentiment_node(state):
        prompt = f"Classify sentiment as exactly one word — positive, neutral, or negative:\n\n{state['message']}\n\nRespond with only the single word."
        resp = router_llm.invoke([HumanMessage(content=prompt)])
        state["sentiment"] = resp.content.strip().lower()
        return state

    def escalation_node(state):
        if state["sentiment"] == "negative" or state["department"] == "Unknown":
            state["escalate"] = True
            state["response"] = "Escalated to a human agent."
        else:
            state["escalate"] = False
        return state

    def rag_node(state):
        retrieved = retriever.get_relevant_documents(state["message"])
        state["retrieved_docs"] = retrieved
        context = "\n".join([d.page_content for d in retrieved])
        prompt = f"""You are ShopUNow's helpful AI assistant.
Use the context below to answer the user's question accurately and concisely.
If the answer isn't in the context, say: "I'm not certain — a human agent will assist you."

Context:
{context}

User Question: {state['message']}

Answer:"""
        resp = answer_llm.invoke([HumanMessage(content=prompt)])
        state["response"] = resp.content
        return state

    def human_node(state):
        state["response"] = "Your query has been escalated to a human support agent. Please provide your contact details and our team will reach out shortly. 🙋"
        return state

    # Build graph
    graph = StateGraph(ChatState)
    graph.add_node("router",     router_node)
    graph.add_node("sentiment",  sentiment_node)
    graph.add_node("escalation", escalation_node)
    graph.add_node("rag",        rag_node)
    graph.add_node("human",      human_node)
    graph.set_entry_point("router")
    graph.add_edge("router",    "sentiment")
    graph.add_edge("sentiment", "escalation")
    graph.add_conditional_edges("escalation", lambda s: "human" if s["escalate"] else "rag", {"human": "human", "rag": "rag"})
    graph.add_edge("human", END)
    graph.add_edge("rag",   END)

    return graph.compile(), retriever

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "stats" not in st.session_state:
    st.session_state.stats = {"total": 0, "escalated": 0, "resolved": 0}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">🛍️ ShopUNow Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Agentic AI · RAG · LangGraph · Enterprise</div>', unsafe_allow_html=True)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # API Key input
    st.markdown("**🔑 Gemini API Key**")
    api_key_input = st.text_input(
        "Enter your key",
        value=GOOGLE_API_KEY,
        type="password",
        label_visibility="collapsed",
        placeholder="AIza..."
    )
    if api_key_input:
        os.environ["GOOGLE_API_KEY"] = api_key_input

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # Session stats
    st.markdown("**📊 Session Stats**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="stat-pill"><div class="stat-num">{st.session_state.stats["total"]}</div><div class="stat-label">Queries</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat-pill"><div class="stat-num">{st.session_state.stats["resolved"]}</div><div class="stat-label">Resolved</div></div>', unsafe_allow_html=True)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # Example queries
    st.markdown("**💡 Try These**")
    examples = [
        "How do I apply for annual leave?",
        "Where do I download my invoice?",
        "My package hasn't arrived!",
        "What's the expense reimbursement policy?",
        "My delivery is late and I'm furious!",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:20]}", use_container_width=True):
            st.session_state["prefill"] = ex

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.stats = {"total": 0, "escalated": 0, "resolved": 0}
        st.rerun()

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.72rem; color:#8B9AB0; text-align:center;">
        Built by <strong style="color:#60A5FA;">Suranjika Sahu</strong><br>
        LangGraph · LangChain · Gemini · FAISS<br><br>
        <a href="https://github.com/suranjika/ShopUNow-AI-Assistant" style="color:#3B82F6;">GitHub</a> ·
        <a href="https://linkedin.com/in/suranjika-sahu" style="color:#3B82F6;">LinkedIn</a>
    </div>
    """, unsafe_allow_html=True)

# ── Main area ──────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🛍️ ShopUNow AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Enterprise Agentic AI · HR · Finance · Billing · Shipping</div>', unsafe_allow_html=True)

# Department badge helper
def dept_badge(dept):
    cls = f"badge-{dept}" if dept in ["HR","Finance","Billing","Shipping"] else "badge-Unknown"
    return f'<span class="dept-badge {cls}">{dept}</span>'

def sentiment_indicator(s):
    icons = {"positive": "😊", "neutral": "😐", "negative": "😤"}
    cls = f"sentiment-{s}" if s in ["positive","neutral","negative"] else "sentiment-neutral"
    return f'<span class="{cls}">{icons.get(s,"❓")} {s}</span>'

# Render chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="msg-user">💬 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        meta = msg.get("meta", {})
        dept  = meta.get("department", "")
        sent  = meta.get("sentiment", "")
        esc   = meta.get("escalated", False)

        badge_html = dept_badge(dept) + sentiment_indicator(sent) if dept else ""

        if esc:
            st.markdown(f'''
            <div class="msg-escalated">
                🚨 <strong>Escalated to Human Agent</strong><br>
                {badge_html}<br><br>
                {msg["content"]}
            </div>''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div class="msg-bot">
                {badge_html}<br><br>
                🤖 {msg["content"]}
            </div>''', unsafe_allow_html=True)

# ── Input area ─────────────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill", "")
user_input = st.text_input(
    "Ask anything...",
    value=prefill,
    placeholder="e.g. How do I apply for annual leave?",
    label_visibility="collapsed",
    key="chat_input"
)

col_send, col_clear = st.columns([5, 1])
with col_send:
    send = st.button("Send →", use_container_width=True)

if send and user_input.strip():
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        st.error("⚠️ Please enter your Gemini API key in the sidebar to use the assistant.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.stats["total"] += 1

        with st.spinner("🤖 Thinking..."):
            agent_app, _ = build_agent()
            if agent_app is None:
                st.error("⚠️ Could not initialise the agent. Please check your API key.")
            else:
                initial_state = ChatState(
                    message=user_input,
                    sentiment="",
                    department="",
                    retrieved_docs=[],
                    response="",
                    escalate=False,
                )
                try:
                    result = agent_app.invoke(initial_state)
                    response   = result["response"]
                    sentiment  = result["sentiment"]
                    department = result["department"]
                    escalated  = result["escalate"]

                    if escalated:
                        st.session_state.stats["escalated"] += 1
                    else:
                        st.session_state.stats["resolved"] += 1

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "meta": {
                            "department": department,
                            "sentiment":  sentiment,
                            "escalated":  escalated,
                        }
                    })
                except Exception as e:
                    st.error(f"❌ Agent error: {str(e)}")

        st.rerun()
