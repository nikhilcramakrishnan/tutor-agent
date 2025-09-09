from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

class AgentState(BaseModel):
    """The state of our RAG agent, using Pydantic for consistency."""
    user_query: str
    original_query: str
    user_sentiment: str = ""
    retrieved_docs: List[Dict[str, Any]] = Field(default_factory=list)
    rewritten_query: Optional[str] = None
    retry_count: int = 0
    final_answer: str = ""
    emotion_state: str = "thinking"
    chat_history: List[BaseMessage] = Field(default_factory=list)
    relevance_score: float = 0.0
    status_updates: List[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True