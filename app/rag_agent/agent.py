import json
import logging
from typing import Dict, Any, List, Optional, Union
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.output_parsers import JsonOutputParser

from app.core.state import AgentState
from app.services.vectordb import ChromaPDFStore
from app.services.sentiment_analyzer import SentimentAnalyzer
from app.services.relevance_checker import RelevanceChecker
from app.rag_agent.nodes import (
    analyze_user_sentiment,
    retrieve_documents,
    check_relevance,
    rewrite_query,
    generate_response,
)

logger = logging.getLogger(__name__)

class RAGAgent:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        """Initializes the RAG Agent with all necessary services."""
        self.llm = ChatOpenAI(model=model_name, temperature=0.1)
        self.vector_store = ChromaPDFStore()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.relevance_checker = RelevanceChecker()
        self.json_output_parser = JsonOutputParser()
        self.graph = self._build_graph()

    def _format_chat_history(self, chat_history: List[Union[Dict[str, str], BaseMessage]]) -> List[BaseMessage]:
        """Converts a list of dicts into a list of LangChain message objects."""
        formatted_history = []
        for message in chat_history:
            if isinstance(message, BaseMessage):
                formatted_history.append(message)
            else:
                role = message.get("role")
                content = message.get("content")
                
                if role == "user":
                    formatted_history.append(HumanMessage(content=content))
                elif role == "agent":
                    formatted_history.append(AIMessage(content=content))
                else:
                    logger.warning(f"Unknown message role: {role}")
        return formatted_history

    def _build_graph(self) -> StateGraph:
        """Builds the LangGraph state machine."""
        workflow = StateGraph(AgentState)

        # Add nodes, passing the necessary services
        workflow.add_node("analyze_user_sentiment", lambda state: analyze_user_sentiment(state, self.sentiment_analyzer))
        workflow.add_node("retrieve_documents", lambda state: retrieve_documents(state, self.vector_store))
        workflow.add_node("check_relevance", lambda state: check_relevance(state, self.relevance_checker))
        workflow.add_node("rewrite_query", lambda state: rewrite_query(state, self.llm))
        workflow.add_node(
            "generate_response",
            lambda state: generate_response(state, self.llm, self.sentiment_analyzer, self.json_output_parser)
        )

        # Define the flow
        workflow.set_entry_point("analyze_user_sentiment")

        # Define edges
        workflow.add_edge("analyze_user_sentiment", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "check_relevance")

        # Self-correction conditional edge
        workflow.add_conditional_edges(
            "check_relevance",
            self._relevance_decision,
            {"sufficient": "generate_response", "insufficient": "rewrite_query"}
        )

        workflow.add_edge("rewrite_query", "retrieve_documents")
        workflow.add_edge("generate_response", END)

        # Compile the graph
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    def _relevance_decision(self, state: AgentState) -> str:
        """Decision function for relevance checking."""
        score = state.relevance_score
        max_retries = 3
        if state.retry_count >= max_retries:
            return "sufficient"
        return "sufficient" if score >= 0.7 else "insufficient"

    async def process_query(self, query: str, chat_history: List[Union[Dict[str, str], BaseMessage]] = None) -> Dict[str, Any]:
        """Main method to process a user query through the agent."""
        if chat_history is None:
            chat_history = []

        formatted_chat_history = self._format_chat_history(chat_history)
        
        initial_state = AgentState(
            user_query=query,
            original_query=query,
            chat_history=formatted_chat_history
        )
        
        config = {"configurable": {"thread_id": "default"}}
        final_state_dict = await self.graph.ainvoke(initial_state, config)
        final_state = AgentState(**final_state_dict)

        return {
            "answer": final_state.final_answer,
            "emotion": final_state.emotion_state,
            "status_updates": final_state.status_updates,
            "retrieved_docs_count": len(final_state.retrieved_docs),
            "retry_count": final_state.retry_count,
            "relevance_score": final_state.relevance_score,
            "used_rewritten_query": final_state.rewritten_query is not None
        }

if __name__ == "__main__":
    async def test_agent():
        agent = RAGAgent()
        
        test_queries = [
            "How is the propagation of neem?",
            "What are plant diseases?",
            "Organic farming benefits"
        ]
        
        for query in test_queries:
            print(f"\n{'='*50}")
            print(f"Testing: {query}")
            print('='*50)
            
            result = await agent.process_query(query)
            print(json.dumps(result, indent=2))
    
    import asyncio
    asyncio.run(test_agent())