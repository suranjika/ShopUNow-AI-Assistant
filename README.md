# 🛍️ ShopUNow AI Assistant

> **An enterprise-grade Agentic AI system** that intelligently handles internal and external queries across HR, Finance, Billing, and Shipping — built with LangGraph, LangChain, Gemini, FAISS, and FastAPI.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-1C3C3C?style=flat&logo=chainlink&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.1+-FF6B35?style=flat)
![Gemini](https://img.shields.io/badge/Gemini-1.5_Flash-4285F4?style=flat&logo=google&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-009688?style=flat)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat&logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## 🎯 What is ShopUNow Assistant?

ShopUNow Assistant is an **Agentic AI application** designed to automate enterprise query resolution at scale. Instead of routing every employee or customer query to a human agent, the system uses a **multi-node LangGraph workflow** to:

- 🔍 **Classify** the query to the correct department (HR, Finance, Billing, Shipping)
- ❤️ **Detect sentiment** (positive / neutral / negative)
- 📚 **Retrieve and answer** using a RAG pipeline backed by a FAISS vector store
- 🚨 **Escalate** negative or unrecognised queries to a human agent
- 🌐 **Serve responses** via a production-ready FastAPI endpoint

This project was built as part of the **Analytics Vidhya — Agentic AI Pioneer Program (2025)**.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────┐
│ Router Node │  ──► Classifies department: HR / Finance / Billing / Shipping / Unknown
└─────────────┘
    │
    ▼
┌────────────────┐
│ Sentiment Node │  ──► Detects: positive / neutral / negative (via Gemini LLM)
└────────────────┘
    │
    ▼
┌──────────────────┐
│ Escalation Node  │  ──► Escalates if: negative sentiment OR unknown department
└──────────────────┘
    │
    ├──── escalate=True  ──► ┌─────────────┐
    │                        │ Human Node  │  ──► Collect user details → Human agent handoff
    │                        └─────────────┘
    │
    └──── escalate=False ──► ┌──────────────┐
                              │   RAG Node   │  ──► FAISS retrieval + Gemini answer generation
                              └──────────────┘
                                    │
                                    ▼
                              FastAPI Response
                          { response, sentiment,
                            department, escalated }
```

### Agent Flow (LangGraph StateGraph)

```
[router] → [sentiment] → [escalation] → [human]  → END
                                      ↘ [rag]    → END
```

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Multi-Agent Orchestration** | LangGraph StateGraph with 5 specialised nodes |
| **RAG Pipeline** | Gemini `embedding-001` + FAISS vector store with semantic search |
| **Sentiment Detection** | LLM-powered sentiment classification per query |
| **Smart Escalation** | Auto-escalates negative or unrecognised queries to human agents |
| **Multi-turn Context** | Stateful `ChatState` TypedDict maintains context across the graph |
| **REST API** | FastAPI `/ask` endpoint with Pydantic request/response validation |
| **4 Department KBs** | HR (internal), Finance (internal), Billing (external), Shipping (external) |
| **48 FAQ Entries** | Embedded and indexed across all departments |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Agent Orchestration** | LangGraph |
| **LLM Framework** | LangChain |
| **LLM Model** | Google Gemini 1.5 Flash |
| **Embeddings** | Google Generative AI `embedding-001` |
| **Vector Store** | FAISS (CPU) |
| **API Layer** | FastAPI + Uvicorn |
| **Data Validation** | Pydantic v2 |
| **Language** | Python 3.10+ |

---

## 📁 Project Structure

```
ShopUNow-AI-Assistant/
│
├── README.md                        # You are here
├── requirements.txt                 # All dependencies
├── .env.example                     # Environment variable template
├── .gitignore
│
├── api/
│   └── main.py                      # FastAPI app — /health + /ask endpoints
│
├── agents/
│   ├── router_node.py               # Department classification logic
│   ├── sentiment_node.py            # Gemini-powered sentiment detection
│   ├── escalation_node.py           # Escalation decision logic
│   ├── rag_node.py                  # FAISS retrieval + Gemini answer generation
│   └── human_node.py                # Human agent handoff node
│
├── rag/
│   ├── knowledge_base.py            # Department FAQ datasets
│   ├── embeddings.py                # Gemini embedding setup
│   └── vectorstore.py               # FAISS index creation & retriever
│
├── graphs/
│   └── workflow.py                  # LangGraph StateGraph definition & compilation
│
├── knowledge_base/
│   └── sample_docs/                 # Sample KB documents (no sensitive data)
│
├── assets/
│   └── architecture.png             # Architecture diagram
│
└── notebooks/
    └── ShopUNow_Demo.ipynb          # Full walkthrough notebook
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/ShopUNow-AI-Assistant.git
cd ShopUNow-AI-Assistant
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

```bash
cp .env.example .env
```

Open `.env` and add your Google API key:

```
GOOGLE_API_KEY=your_gemini_api_key_here
```

Get a free Gemini API key at: [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

### 5. Run the FastAPI Server

```bash
uvicorn api.main:api --reload --host 0.0.0.0 --port 8000
```

### 6. Test the API

Open your browser at **http://localhost:8000/docs** for the interactive Swagger UI.

Or use `curl`:

```bash
# Standard query
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I apply for annual leave?"}'

# Escalation query (negative sentiment)
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "My delivery is late again, this is completely unacceptable!",
    "name": "John Doe",
    "phone": "+62 812 3456 7890",
    "email": "john@example.com"
  }'
```

---

## 📡 API Reference

### `GET /health`
Health check endpoint.

**Response:**
```json
{ "ok": true, "departments": ["HR", "Finance", "Billing", "Shipping"] }
```

---

### `POST /ask`
Submit a query to the Agentic AI assistant.

**Request Body:**
```json
{
  "query": "string (required)",
  "name":  "string (optional — required if escalated)",
  "phone": "string (optional — required if escalated)",
  "email": "string (optional — required if escalated)"
}
```

**Response:**
```json
{
  "response":       "string",
  "sentiment":      "positive | neutral | negative",
  "department":     "HR | Finance | Billing | Shipping | Unknown",
  "escalated":      "boolean",
  "need_details":   "boolean",
  "required_fields": ["name", "phone", "email"],
  "echo_details":   { "name": "...", "phone": "...", "email": "..." }
}
```

---

## 💬 Example Queries & Outputs

| Query | Department | Sentiment | Escalated | Response |
|---|---|---|---|---|
| *"How do I apply for annual leave?"* | HR | positive | ❌ | RAG answer from HR KB |
| *"Where do I download my invoice?"* | Billing | positive | ❌ | RAG answer from Billing KB |
| *"The delivery is late again, this is frustrating!"* | Shipping | negative | ✅ | Human agent handoff |
| *"Help with my expense reimbursement"* | Finance | positive | ❌ | RAG answer from Finance KB |
| *"thanks"* | Unknown | positive | ✅ | Human agent handoff |

---

## 🧠 How the RAG Pipeline Works

1. **Knowledge Base** — 48 FAQ entries across 4 departments are stored as `Document` objects with department metadata
2. **Chunking** — `RecursiveCharacterTextSplitter` splits documents (chunk_size=200, overlap=20)
3. **Embedding** — Google Gemini `embedding-001` generates dense vector embeddings
4. **Indexing** — FAISS stores vectors for fast approximate nearest-neighbour search
5. **Retrieval** — Top-2 most relevant chunks are retrieved per query (`k=2`)
6. **Generation** — Gemini 1.5 Flash generates a grounded answer from the retrieved context

---

## 🗺️ LangGraph State Schema

```python
class ChatState(TypedDict):
    message:        str            # User's input query
    sentiment:      str            # positive / neutral / negative
    department:     str            # HR / Finance / Billing / Shipping / Unknown
    retrieved_docs: List[Document] # FAISS retrieval results
    response:       str            # Final assistant response
    escalate:       bool           # Whether to route to human agent
```

---

## 🔮 Stretch Goals & Future Improvements

- [ ] Add **conversation memory** for true multi-turn dialogue
- [ ] Integrate **Streamlit / Gradio** frontend for live demo
- [ ] Deploy on **Hugging Face Spaces** for public access
- [ ] Add **LangSmith tracing** for agent observability
- [ ] Replace keyword-based router with **LLM-powered classification**
- [ ] Add **confidence scoring** to RAG responses
- [ ] Support **multilingual queries** (Bahasa Indonesia)
- [ ] Containerise with **Docker** for production deployment

---

## 📚 Key Learnings

- LangGraph's `StateGraph` is a powerful pattern for orchestrating multi-step agentic workflows — state flows cleanly between nodes
- FAISS with Gemini embeddings provides surprisingly accurate semantic retrieval even on small knowledge bases
- Separating routing logic from answer generation makes the system modular and easier to debug
- FastAPI's Pydantic models enforce a clean contract between the AI backend and any frontend/consumer
- Escalation logic needs to be a first-class concern in enterprise AI — not an afterthought

---

## 👩‍💻 Author

**Suranjika Sahu**
Data & Analytics Leader | AI/GenAI | GCP | Jakarta, Indonesia

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat&logo=linkedin)](https://linkedin.com/in/YOUR_LINKEDIN)
[![Email](https://img.shields.io/badge/Email-suranjika.sahu@gmail.com-D14836?style=flat&logo=gmail&logoColor=white)](mailto:suranjika.sahu@gmail.com)

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

*Built as part of the Analytics Vidhya — Agentic AI Pioneer Program, 2025*
