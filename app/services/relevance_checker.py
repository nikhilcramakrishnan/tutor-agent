from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class RelevanceChecker:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        """Initialize relevance checker."""
        self.llm = ChatOpenAI(model=model_name, temperature=0)
    
    def check_relevance(self, query: str, documents: List[Dict[str, Any]]) -> float:
        """
        Check if the retrieved documents are relevant to answer the user's query.
        
        Returns a relevance score between 0 and 1.
        """
        if not documents:
            logger.warning("No documents to check relevance")
            return 0.0
        
        # Prepare document summaries for evaluation
        doc_summaries = []
        for i, doc in enumerate(documents[:3]):  # Check top 3 docs only
            content_snippet = doc["content"][:200]  # First 200 chars
            page = doc.get("metadata", {}).get("page", "Unknown")
            doc_summaries.append(f"Doc {i+1} (Page {page}): {content_snippet}...")
        
        prompt = f"""
        Evaluate if these retrieved documents contain information relevant to answering the user's query.
        
        User Query: "{query}"
        
        Retrieved Documents:
        {chr(10).join(doc_summaries)}
        
        Scoring Guidelines:
        - Score 1.0: Documents directly answer the query with specific, relevant information
        - Score 0.8: Documents contain mostly relevant information with good detail
        - Score 0.6: Documents are somewhat relevant but lack specific details needed
        - Score 0.4: Documents are tangentially related but don't really answer the query
        - Score 0.2: Documents contain minimal relevant information
        - Score 0.0: Documents are completely irrelevant to the query
        
        Consider:
        1. Do the documents contain specific information to answer the query?
        2. Are the documents on the right topic?
        3. Is there enough detail to provide a helpful answer?
        
        Respond with only a number between 0.0 and 1.0 (e.g., 0.8)
        """
        
        try:
            response = self.llm.invoke([SystemMessage(content=prompt)]).content.strip()
            
            # Extract numerical score
            import re
            score_match = re.search(r'0?\.\d+|[01]\.0?', response)
            if score_match:
                score = float(score_match.group())
                # Ensure score is in valid range
                score = max(0.0, min(1.0, score))
            else:
                logger.warning(f"Could not parse relevance score from: {response}")
                score = 0.5  # Default to middle score if parsing fails
            
            logger.info(f"Relevance score: {score}")
            return score
            
        except Exception as e:
            logger.error(f"Error checking relevance: {e}")
            return 0.5  # Default to moderate relevance on error
    
    def explain_relevance_issues(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """
        Provide explanation of why documents might not be relevant.
        Useful for debugging and improving the system.
        """
        if not documents:
            return "No documents were retrieved."
        
        doc_summaries = []
        for i, doc in enumerate(documents[:2]):
            content_snippet = doc["content"][:150]
            doc_summaries.append(f"Doc {i+1}: {content_snippet}...")
        
        prompt = f"""
        Explain why these documents might not be sufficient to answer the user's query.
        
        User Query: "{query}"
        
        Documents:
        {chr(10).join(doc_summaries)}
        
        Provide a brief explanation of:
        1. What information is missing
        2. What would make the documents more relevant
        3. Suggested improvements for search
        
        Keep response concise (2-3 sentences).
        """
        
        try:
            response = self.llm.invoke([SystemMessage(content=prompt)]).content.strip()
            return response
        except Exception as e:
            logger.error(f"Error explaining relevance: {e}")
            return "Unable to determine specific relevance issues."