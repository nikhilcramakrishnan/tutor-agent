import logging
from typing import Dict, Any, List, Union
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from pydantic import ValidationError

from app.core.state import AgentState
from app.services.vectordb import ChromaPDFStore
from app.services.sentiment_analyzer import SentimentAnalyzer
from app.services.relevance_checker import RelevanceChecker
from app.core.schemas import EmotionState

logger = logging.getLogger(__name__)

def analyze_user_sentiment(state: AgentState, sentiment_analyzer: SentimentAnalyzer) -> Dict[str, Any]:
    """Analyzes user sentiment."""
    logger.info("🧠 Analyzing user sentiment...")
    sentiment = sentiment_analyzer.analyze(state.user_query)
    logger.info(f"Detected sentiment: {sentiment}")
    return {
        "user_sentiment": sentiment,
        "status_updates": state.status_updates + ["analyzing_sentiment"]
    }

def retrieve_documents(state: AgentState, vector_store: ChromaPDFStore) -> Dict[str, Any]:
    """Retrieves relevant documents from vector store."""
    logger.info("📚 Retrieving documents...")
    query = state.rewritten_query or state.original_query
    logger.info(f"Searching documents for query: '{query}'")
    docs = vector_store.similarity_search(query, k=4)
    retrieved_docs = [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "page": doc.metadata.get("page"),
            "chunk_id": doc.metadata.get("chunk_id")
        }
        for doc in docs
    ]
    logger.info(f"Retrieved {len(docs)} documents.")
    return {
        "retrieved_docs": retrieved_docs,
        "status_updates": state.status_updates + ["searching_knowledge"]
    }

def check_relevance(state: AgentState, relevance_checker: RelevanceChecker) -> Dict[str, Any]:
    """Checks if retrieved documents are relevant to answer the query."""
    logger.info("🔍 Checking document relevance...")
    relevance_score = relevance_checker.check_relevance(
        query=state.user_query,
        documents=state.retrieved_docs
    )
    logger.info(f"Relevance score: {relevance_score}")
    
    return {
        "relevance_score": relevance_score,
        "retry_count": state.retry_count + 1,
        "status_updates": state.status_updates + ["evaluating_sources"]
    }

def rewrite_query(state: AgentState, llm: ChatOpenAI) -> Dict[str, Any]:
    """Rewrites the query for better document retrieval."""
    logger.info("✏️ Rewriting query for better results...")
    if state.retry_count >= 3:
        logger.warning("Max retries reached, proceeding with available docs")
        return {"relevance_score": 0.8}
    
    # Contextualize with chat history for better query rewriting
    chat_context = "\n".join([f"{msg.type}: {msg.content}" for msg in state.chat_history])
    
    # Use a two-step prompt to first identify the subject, then rewrite the query
    prompt = f"""
    The original query didn't retrieve relevant documents. Rewrite it for better search results.
    
    **Chat History for Context:**
    ---
    {chat_context}
    ---
    
    Original Query: "{state.original_query}"
    
    Instructions:
    1.  Based on the chat history and original query, identify the main subject of the conversation.
    2.  Rewrite the original query, making it more specific by including the subject you identified.
    
    Example:
    Chat History: "user: tell me about Neem\nagent: Neem is a tree...\nuser: how is it propagated?"
    Output: "How is the propagation of Neem done?"
    
    Return only the rewritten query, no explanation.
    """
    try:
        rewritten = llm.invoke([SystemMessage(content=prompt)]).content.strip().strip('"\'')
    except Exception as e:
        logger.error(f"Error rewriting query: {e}")
        rewritten = " ".join(state.original_query.split()[:3])
    
    return {
        "rewritten_query": rewritten,
        "status_updates": state.status_updates + ["refining_search"]
    }

def generate_response(
    state: AgentState,
    llm: ChatOpenAI,
    sentiment_analyzer: SentimentAnalyzer,
    json_output_parser: JsonOutputParser
) -> Dict[str, Any]:
    """Generates the final response with emotion state."""
    logger.info("🎭 Generating final response...")
    retrieved_docs = state.retrieved_docs
    if not retrieved_docs:
        doc_context = "No relevant documents were found in the knowledge base."
    else:
        doc_context = "\n\n".join(
            f"[Page {doc.get('page', 'N/A')}] {doc['content']}" for doc in retrieved_docs
        )
    
    tone = sentiment_analyzer.get_response_tone(state.user_sentiment)
    
    prompt_template = """
    You are an expert AI tutor. Your task is to answer the user's query based *only* on the provided context.
    
    **Context from Knowledge Base:**
    ---
    {context}
    ---
    
    **User's Query:** "{query}"
    
    **Instructions:**
    1.  Analyze the context and the user's query.
    2.  If the context contains a direct answer, provide it in a helpful, {tone} tone.
    3.  If the context does **not** contain the answer, you MUST state that you could not find the information in the available documents and do not provide an answer.
    4.  Your entire output must be a single, valid JSON object with two keys: "answer" (your response) and "emotion" (your delivery emotion, e.g., "explaining", "confused").
    
    {format_instructions}
    """
    prompt = prompt_template.format(
        context=doc_context,
        query=state.original_query,
        tone=tone,
        format_instructions=json_output_parser.get_format_instructions()
    )
    
    try:
        chain = llm | json_output_parser
        response_data = chain.invoke(prompt)
        answer = response_data.get("answer")
        emotion = response_data.get("emotion")
        
        # Validate the emotion against the enum to prevent crashes
        try:
            EmotionState(emotion)
        except ValueError:
            logger.warning(f"Invalid emotion '{emotion}' from LLM. Falling back to 'explaining'.")
            emotion = "explaining"
        
    except Exception as e:
        logger.error(f"Fatal error during response generation: {e}")
        logger.debug(f"Failing prompt:\n{prompt}")
        answer = "I'm having trouble formulating a response right now. Please try rephrasing your question."
        emotion = "confused"
    
    return {
        "final_answer": answer,
        "emotion_state": emotion,
        "status_updates": state.status_updates + ["generating_response"]
    }