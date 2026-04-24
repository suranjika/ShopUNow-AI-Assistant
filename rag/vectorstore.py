"""
vectorstore.py
--------------
Builds the FAISS vector store from the department knowledge base
using Google Gemini embeddings.
"""

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from rag.knowledge_base import DATASETS


def build_vectorstore(k: int = 2) -> tuple[FAISS, object]:
    """
    Embed all KB documents and build a FAISS index.

    Args:
        k: Number of documents to retrieve per query.

    Returns:
        (vectorstore, retriever) tuple.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Build Document objects with department metadata
    docs = []
    for dept, entries in DATASETS.items():
        for entry in entries:
            content = entry["q"] + " " + entry["a"]
            docs.append(Document(
                page_content=content,
                metadata={
                    "department": dept,
                    "audience": entry["audience"],
                }
            ))

    # Chunk documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    split_docs = splitter.split_documents(docs)

    # Build FAISS index
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    return vectorstore, retriever
