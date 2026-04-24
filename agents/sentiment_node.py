"""
sentiment_node.py
-----------------
Uses Gemini LLM to classify the sentiment of the user message.
Result: "positive" | "neutral" | "negative"
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from graphs.state import ChatState


# Shared LLM instance (low temperature for classification tasks)
_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)


def sentiment_node(state: ChatState) -> ChatState:
    """
    Classify sentiment of the user's message via Gemini.
    Sets state['sentiment'] to: positive | neutral | negative
    """
    prompt = (
        "Classify the sentiment of the following message as exactly one word: "
        "positive, neutral, or negative.\n\n"
        f"Message: {state['message']}\n\n"
        "Respond with only the single word."
    )
    response = _llm.invoke([HumanMessage(content=prompt)])
    state["sentiment"] = response.content.strip().lower()
    return state
