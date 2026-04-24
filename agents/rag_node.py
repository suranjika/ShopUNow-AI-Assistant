"""
rag_node.py
-----------
Retrieves relevant documents from the FAISS vector store
and generates a grounded answer using Gemini.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from graphs.state import ChatState


# Shared answer LLM
_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)


def build_rag_node(retriever):
    """
    Factory function — returns a rag_node function bound to the given retriever.
    Call this once at startup after building the FAISS index.
    """

    def rag_node(state: ChatState) -> ChatState:
        """
        1. Retrieve top-k relevant documents from FAISS
        2. Build a grounded prompt with context
        3. Generate answer via Gemini
        """
        docs = retriever.get_relevant_documents(state["message"])
        state["retrieved_docs"] = docs

        context = "\n".join([d.page_content for d in docs])

        prompt = f"""You are ShopUNow's helpful AI assistant.
Use the context below to answer the user's question accurately and concisely.
If the answer isn't in the context, say: "I'm not certain — a human agent will assist you."

Context:
{context}

User Question: {state['message']}

Answer:"""

        response = _llm.invoke([HumanMessage(content=prompt)])
        state["response"] = response.content
        return state

    return rag_node
