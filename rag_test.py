# File: backend/rag_test.py

import json
import asyncio
from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from app.rag_agent.agent import RAGAgent

def format_chat_history(chat_history: List[Dict[str, str]]) -> List[BaseMessage]:
    """Converts a list of dicts into a list of LangChain message objects."""
    formatted_history = []
    for message in chat_history:
        role = message.get("role")
        content = message.get("content")
        
        if role == "user":
            formatted_history.append(HumanMessage(content=content))
        elif role == "agent":
            formatted_history.append(AIMessage(content=content))
        else:
            print(f"Unknown message role: {role}")
    return formatted_history


async def test_rag(query: str, chat_history: List[Dict[str, str]] = None):
    """
    Tests the RAG Agent with a single query.
    The agent now encapsulates the entire RAG pipeline.
    """
    if chat_history is None:
        chat_history = []

    print("\n--- Starting RAG Test ---")
    
    # Initialize the refactored agent
    agent = RAGAgent()
    
    # Format chat history before passing to the agent
    formatted_history = format_chat_history(chat_history)
    
    # Process the query through the agent's main method
    result = await agent.process_query(query, formatted_history)
    
    print("\n--- Final Answer ---")
    print(json.dumps(result, indent=2))
    print("\n--- Metadata ---")
    print(f"Retrieved docs count: {result.get('retrieved_docs_count')}")
    print(f"Relevance score: {result.get('relevance_score'):.2f}")
    print(f"Emotion: {result.get('emotion')}")


if __name__ == "__main__":
    async def run_test():
        await test_rag("How is the propogation of neem?")
    
    asyncio.run(run_test())