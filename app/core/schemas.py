# File: backend/app/core/schemas.py

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime
from langchain_core.messages import BaseMessage

class EmotionState(str, Enum):
    """Available emotion states for the mascot."""
    HAPPY = "happy"
    EXPLAINING = "explaining"
    THINKING = "thinking"
    CONFUSED = "confused"
    CURIOUS = "curious"
    NEUTRAL = "neutral"
    ENTHUSIASTIC = "enthusiastic"
    APOLOGETIC = "apologetic"
    FRUSTRATED = "frustrated"

class StatusType(str, Enum):
    """Status update types for real-time feedback."""
    ANALYZING_SENTIMENT = "analyzing_sentiment"
    SEARCHING_KNOWLEDGE = "searching_knowledge"
    EVALUATING_SOURCES = "evaluating_sources"
    REFINING_SEARCH = "refining_search"
    GENERATING_RESPONSE = "generating_response"
    COMPLETE = "complete"

class LipSyncFrame(BaseModel):
    """Individual frame of lip-sync data."""
    time: float = Field(description="Time in seconds")
    value: str = Field(description="Viseme/mouth shape identifier")

class StatusUpdate(BaseModel):
    """Real-time status update."""
    status: StatusType
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)

class QueryRequest(BaseModel):
    """Request for processing a user query."""
    query: str = Field(description="User's question or input")
    session_id: Optional[str] = Field(default=None, description="Session identifier for context")
    chat_history: Optional[List[Dict[str, str]]] = Field(default=None, description="Previous conversation")
    include_audio: bool = Field(default=True, description="Whether to generate TTS audio")
    include_lip_sync: bool = Field(default=True, description="Whether to generate lip-sync data")

class STTRequest(BaseModel):
    """Request for speech-to-text transcription."""
    audio_duration: Optional[float] = Field(default=5.0, description="Recording duration in seconds")
    audio_file: Optional[str] = Field(default=None, description="Base64 encoded audio file")

class AgentResponse(BaseModel):
    """Basic agent response without audio."""
    answer: str = Field(description="The agent's text response")
    emotion: EmotionState = Field(description="Emotion state for mascot animation")
    confidence: float = Field(default=0.8, description="Response confidence score")
    sources_used: List[str] = Field(default=[], description="Sources referenced in response")

class ComprehensiveResponse(BaseModel):
    """Complete response including audio and lip-sync data."""
    answer_text: str = Field(description="The agent's text response")
    emotion_state: EmotionState = Field(description="Emotion for mascot animation")
    tts_audio: Optional[str] = Field(default=None, description="Base64 encoded audio data")
    lip_sync_data: Optional[List[LipSyncFrame]] = Field(default=None, description="Lip-sync animation data")
    status_updates: List[str] = Field(default=[], description="Processing status updates")
    metadata: Dict[str, Any] = Field(default={}, description="Additional processing metadata")

class SessionInfo(BaseModel):
    """Information about an active processing session."""
    session_id: str
    status: str
    start_time: datetime
    updates: List[StatusUpdate]
    elapsed_time: float

class HealthStatus(BaseModel):
    """Health status of various system components."""
    api: str
    agent: str
    vector_store: str
    llm: str
    timestamp: datetime

class ErrorResponse(BaseModel):
    """Standardized error response."""
    error: str
    detail: str
    timestamp: datetime
    session_id: Optional[str] = None

class RelevanceCheckRequest(BaseModel):
    """For testing relevance checking."""
    query: str
    documents: List[Dict[str, Any]]

class RelevanceCheckResponse(BaseModel):
    """Response from relevance checking."""
    relevance_score: float
    explanation: str
    recommendation: str  # "sufficient" or "insufficient"

class AgentConfig(BaseModel):
    """Configuration for the agent."""
    max_retries: int = Field(default=3, description="Maximum query rewrite attempts")
    relevance_threshold: float = Field(default=0.7, description="Minimum relevance score")
    retrieval_k: int = Field(default=4, description="Number of documents to retrieve")
    
class TTSConfig(BaseModel):
    """Configuration for text-to-speech."""
    voice_id: Optional[str] = None
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    include_lip_sync: bool = Field(default=True)