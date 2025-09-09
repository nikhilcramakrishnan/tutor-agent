# File: backend/app/api/main.py

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import json
import logging
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# Corrected import path for RAGAgent
from app.rag_agent.agent import RAGAgent
from app.services.stt_service import STTService
from app.services.tts_service import TTSService
# Corrected import path for schemas
from app.core.schemas import (
    QueryRequest,
    AgentResponse,
    StatusUpdate,
    STTRequest,
    ComprehensiveResponse
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Agent API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services ONLY ONCE outside the request handler
agent = RAGAgent()
stt_service = STTService()
tts_service = TTSService()

# Store active sessions for real-time updates
active_sessions: Dict[str, Dict] = {}


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "RAG Agent API is running", "timestamp": datetime.now().isoformat()}

@app.post("/agent/query", response_model=ComprehensiveResponse)
async def process_agent_query(request: QueryRequest):
    """
    Main endpoint for processing queries through the LangGraph agent.
    Returns comprehensive response with audio and lip-sync data.
    """
    session_id = request.session_id or "default"

    try:
        logger.info(f"Processing query for session {session_id}: {request.query}")

        # Initialize session tracking
        active_sessions[session_id] = {
            "status": "processing",
            "start_time": datetime.now(),
            "updates": []
        }

        # Pass the raw chat history to the agent for formatting and processing
        agent_result = await agent.process_query(
            query=request.query,
            chat_history=request.chat_history or []
        )

        # Generate TTS and lip-sync data if requested
        audio_data = None
        lip_sync_data = None

        if request.include_audio:
            audio_data = await tts_service.generate_speech(agent_result["answer"])

            if request.include_lip_sync and audio_data and audio_data.get("success"):
                lip_sync_result = await tts_service.generate_lip_sync(audio_data["audio_path"])
                lip_sync_data = lip_sync_result.get("visemes", [])

        # Prepare comprehensive response
        response = ComprehensiveResponse(
            answer_text=agent_result["answer"],
            emotion_state=agent_result["emotion"],
            tts_audio=audio_data.get("audio_base64") if audio_data else None,
            lip_sync_data=lip_sync_data,
            status_updates=agent_result["status_updates"],
            metadata={
                "retrieved_docs_count": agent_result["retrieved_docs_count"],
                "retry_count": agent_result["retry_count"],
                "relevance_score": agent_result["relevance_score"],
                "used_rewritten_query": agent_result["used_rewritten_query"],
                "processing_time": (datetime.now() - active_sessions[session_id]["start_time"]).total_seconds()
            }
        )

        # Clean up session
        if session_id in active_sessions:
            del active_sessions[session_id]

        return response

    except Exception as e:
        logger.error(f"Error processing query: {e}")

        # Clean up session
        if session_id in active_sessions:
            del active_sessions[session_id]

        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/agent/simple", response_model=AgentResponse)
async def process_simple_query(request: QueryRequest):
    """
    Simplified endpoint that returns just text response without audio.
    Useful for testing and development.
    """
    try:
        logger.info(f"Processing simple query: {request.query}")

        formatted_chat_history = []
        for message in request.chat_history or []:
            if message.get("role") == "user":
                formatted_chat_history.append(HumanMessage(content=message.get("content")))
            elif message.get("role") == "agent":
                formatted_chat_history.append(AIMessage(content=message.get("content")))

        agent_result = await agent.process_query(request.query, formatted_chat_history)

        return AgentResponse(
            answer=agent_result["answer"],
            emotion=agent_result["emotion"],
            confidence=agent_result["relevance_score"],
            sources_used=[f"Page {doc.get('page', 'N/A')}" for doc in agent_result.get("retrieved_docs", [])]
        )

    except Exception as e:
        logger.error(f"Error in simple query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stt/transcribe")
async def transcribe_audio(request: STTRequest):
    """Transcribe audio using the STT service."""
    try:
        if request.audio_duration:
            # Live transcription
            transcription = stt_service.transcribe(duration=request.audio_duration)
        else:
            transcription = "Please specify audio_duration for live transcription"

        return {
            "transcription": transcription,
            "confidence": 0.95,
            "timestamp": datetime.now().isoformat(),
            "duration": request.audio_duration
        }

    except Exception as e:
        logger.error(f"STT Error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.get("/agent/status/{session_id}")
async def get_session_status(session_id: str):
    """Get current status of a processing session."""
    if session_id in active_sessions:
        session = active_sessions[session_id]
        return {
            "session_id": session_id,
            "status": session["status"],
            "start_time": session["start_time"].isoformat(),
            "updates": session["updates"],
            "elapsed_time": (datetime.now() - session["start_time"]).total_seconds()
        }
    else:
        return {"session_id": session_id, "status": "not_found"}

@app.get("/health")
async def health_check():
    """Comprehensive health check of all services."""
    health_status = {
        "api": "healthy",
        "agent": "unknown",
        "vector_store": "unknown",
        "llm": "unknown",
        "timestamp": datetime.now().isoformat()
    }

    try:
        test_result = await agent.process_query("Hello")
        health_status["agent"] = "healthy" if test_result.get("answer") else "error"
    except Exception as e:
        health_status["agent"] = f"error: {str(e)}"

    try:
        docs = agent.vector_store.similarity_search("test", k=1)
        health_status["vector_store"] = "healthy" if docs else "no_data"
    except Exception as e:
        health_status["vector_store"] = f"error: {str(e)}"

    try:
        response = agent.llm.invoke("Say 'OK' if you can respond").content
        health_status["llm"] = "healthy" if "OK" in response else "error"
    except Exception as e:
        health_status["llm"] = f"error: {str(e)}"

    return health_status

@app.post("/debug/relevance")
async def debug_relevance_check(query: str, k: int = 4):
    """Debug endpoint to test document retrieval and relevance checking."""
    try:
        docs = agent.vector_store.similarity_search(query, k=k)
        doc_data = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "page": doc.metadata.get("page"),
                "chunk_id": doc.metadata.get("chunk_id")
            }
            for doc in docs
        ]

        relevance_score = agent.relevance_checker.check_relevance(query, doc_data)

        return {
            "query": query,
            "documents_found": len(doc_data),
            "relevance_score": relevance_score,
            "recommendation": "sufficient" if relevance_score >= 0.7 else "insufficient",
            "documents": [
                {
                    "page": doc["page"],
                    "chunk_id": doc["chunk_id"],
                    "preview": doc["content"][:150] + "..."
                }
                for doc in doc_data
            ]
        }

    except Exception as e:
        logger.error(f"Debug relevance error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for better error responses."""
    logger.error(f"Unhandled exception: {exc}")
    return {
        "error": "Internal server error",
        "detail": str(exc),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")