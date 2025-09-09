from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
import json
import logging

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        """Initialize sentiment analyzer with a lightweight model."""
        self.llm = ChatOpenAI(model=model_name, temperature=0)
    
    def analyze(self, user_input: str) -> str:
        """
        Analyze user sentiment to help the agent respond appropriately.
        
        Returns one of: confused, curious, frustrated, excited, neutral
        """
        prompt = f"""
        Analyze the sentiment and emotional state of this user input.
        
        User input: "{user_input}"
        
        Determine the user's primary emotional state from these options:
        - confused: User seems lost, uncertain, or asks for clarification
        - curious: User is eager to learn, asks exploratory questions  
        - frustrated: User seems annoyed, impatient, or repeating failed queries
        - excited: User shows enthusiasm, exclamation points, positive energy
        - neutral: Standard informational request, no strong emotion
        
        Respond with only the sentiment word, no explanation.
        """
        
        try:
            response = self.llm.invoke([SystemMessage(content=prompt)]).content.strip().lower()
            
            # Validate response
            valid_sentiments = ["confused", "curious", "frustrated", "excited", "neutral"]
            if response in valid_sentiments:
                return response
            else:
                logger.warning(f"Invalid sentiment returned: {response}, defaulting to neutral")
                return "neutral"
                
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return "neutral"
    
    def get_response_tone(self, sentiment: str) -> str:
        """Get appropriate response tone based on user sentiment."""
        tone_mapping = {
            "confused": "patient and clear, breaking down complex topics step by step",
            "curious": "enthusiastic and detailed, encouraging exploration",
            "frustrated": "calm and reassuring, focusing on solving their problem quickly",
            "excited": "matching their energy while staying informative",
            "neutral": "friendly and professional"
        }
        
        return tone_mapping.get(sentiment, "friendly and professional")