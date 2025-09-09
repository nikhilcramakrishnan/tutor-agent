import asyncio
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

# Updated import path for the refactored agent
from app.rag_agent.agent import RAGAgent
from app.services.stt_service import STTService
from app.services.tts_service import TTSService

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

class IntegrationTester:
    def __init__(self):
        """Initialize integration tester."""
        self.agent = None
        self.stt_service = None
        self.tts_service = None
    
    async def setup_services(self):
        """Initialize all services."""
        print("🚀 Initializing services...")
        
        try:
            self.agent = RAGAgent()
            print("✅ LangGraph Agent initialized")
        except Exception as e:
            print(f"❌ Agent initialization failed: {e}")
            return False
        
        try:
            self.stt_service = STTService()
            print("✅ STT Service initialized")
        except Exception as e:
            print(f"❌ STT Service initialization failed: {e}")
        
        try:
            self.tts_service = TTSService()
            print("✅ TTS Service initialized")
        except Exception as e:
            print(f"❌ TTS Service initialization failed: {e}")
        
        return True
    
    async def test_basic_rag(self):
        """Test basic RAG functionality."""
        print("\n📚 Testing Basic RAG...")
        
        test_queries = [
            "How is the propagation of neem?",
            "What are the benefits of organic farming?", 
            "Tell me about plant diseases",
            "How do I care for tomatoes?"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            try:
                result = await self.agent.process_query(query)
                print(f"Answer: {result['answer'][:100]}...")
                print(f"Emotion: {result['emotion']}")
                print(f"Status Updates: {' → '.join(result['status_updates'])}")
                print(f"Docs Retrieved: {result['retrieved_docs_count']}")
                print(f"Relevance Score: {result['relevance_score']:.2f}")
                if result['used_rewritten_query']:
                    print("🔄 Query was rewritten for better results")
            except Exception as e:
                print(f"❌ Error: {e}")
    
    async def test_self_correction(self):
        """Test the self-correction mechanism."""
        print("\n🔄 Testing Self-Correction...")
        
        # These queries are designed to potentially need query rewriting
        correction_queries = [
            "AI stuff",  # Very vague
            "How to make things grow better?",  # Ambiguous
            "Problems with stuff",  # Very generic
            "Help me with gardening issues"  # Could be too broad
        ]
        
        for query in correction_queries:
            print(f"\nCorrection Test: '{query}'")
            try:
                result = await self.agent.process_query(query)
                print(f"Answer: {result['answer'][:100]}...")
                print(f"Retry Count: {result['retry_count']}")
                print(f"Final Relevance: {result['relevance_score']:.2f}")
                
                if result['retry_count'] > 0:
                    print("✅ Self-correction triggered successfully")
                else:
                    print("ℹ️ No correction needed - documents were relevant")
                    
            except Exception as e:
                print(f"❌ Correction Error: {e}")
    
    def test_stt(self):
        """Test STT service (interactive)."""
        if not self.stt_service:
            print("❌ STT Service not available")
            return None
        
        print("\n🎤 Testing STT Service...")
        print("This will record audio for 3 seconds. Speak clearly!")
        
        try:
            input("Press Enter to start recording...")
            transcription = self.stt_service.transcribe(duration=3)
            print(f"Transcribed: '{transcription}'")
            
            if transcription and transcription.strip():
                print("✅ STT working correctly")
                return transcription
            else:
                print("⚠️ No speech detected or transcription empty")
                return None
                
        except Exception as e:
            print(f"❌ STT Error: {e}")
            return None
    
    async def test_tts_pipeline(self):
        """Test TTS and lip-sync generation."""
        if not self.tts_service:
            print("❌ TTS Service not available")
            return
        
        print("\n🎵 Testing TTS Pipeline...")
        
        test_text = "Hello! I'm your AI tutor, and I'm excited to help you learn about plants today."
        
        try:
            result = await self.tts_service.test_tts_pipeline(test_text)
            
            if result.get("pipeline_success"):
                print("✅ TTS Pipeline working correctly")
                print(f"Audio duration: {result['audio_test'].get('duration', 0):.2f}s")
                print(f"Lip-sync frames: {len(result['lip_sync_test'].get('visemes', []))}")
                
                # Show sample lip-sync data
                visemes = result['lip_sync_test'].get('visemes', [])
                if visemes:
                    print("Sample lip-sync frames:")
                    for frame in visemes[:3]:
                        print(f"  {frame['time']:.1f}s: {frame['value']}")
            else:
                print("⚠️ TTS Pipeline has issues")
                print(f"Error: {result.get('error', 'Unknown')}")
                
        except Exception as e:
            print(f"❌ TTS Pipeline Error: {e}")
    
    async def test_full_pipeline(self):
        """Test the complete pipeline from query to final response."""
        print("\n🔄 Testing Full Pipeline...")
        
        # Option 1: Use STT for input
        print("Choose input method:")
        print("1. Voice input (STT)")
        print("2. Text input")
        
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1" and self.stt_service:
            user_input = self.test_stt()
            if not user_input:
                user_input = "How do plants grow?"  # Fallback
        else:
            user_input = input("Enter your question: ").strip() or "How do plants grow?"
        
        print(f"\nProcessing: '{user_input}'")
        print("-" * 50)
        
        try:
            # Process through agent
            agent_result = await self.agent.process_query(user_input)
            
            print("📋 Agent Processing Results:")
            print(f"📝 Answer: {agent_result['answer']}")
            print(f"😊 Emotion: {agent_result['emotion']}")
            print(f"🔄 Processing Steps: {' → '.join(agent_result['status_updates'])}")
            print(f"📄 Documents Retrieved: {agent_result['retrieved_docs_count']}")
            print(f"🎯 Relevance Score: {agent_result['relevance_score']:.2f}")
            if agent_result['used_rewritten_query']:
                print("🔄 Query was automatically improved")
            
            # Generate TTS and lip-sync
            if self.tts_service:
                print(f"\n🎵 Generating speech and lip-sync...")
                audio_result = await self.tts_service.generate_speech(agent_result['answer'])
                
                if audio_result.get("success"):
                    print("✅ Speech generation successful")
                    
                    # Generate lip-sync
                    lip_sync_result = await self.tts_service.generate_lip_sync(audio_result["audio_path"])
                    if lip_sync_result.get("success"):
                        print("✅ Lip-sync generation successful")
                        print(f"👄 Viseme frames: {len(lip_sync_result['visemes'])}")
                    
                    # Simulate the final API response format
                    final_response = {
                        "answer_text": agent_result['answer'],
                        "emotion_state": agent_result['emotion'],
                        "tts_audio": audio_result.get("audio_base64", "")[:-20] + "..." if audio_result.get("audio_base64") else "",  # Truncate for display
                        "lip_sync_data": lip_sync_result.get("visemes", []),
                        "status_updates": agent_result['status_updates'],
                        "metadata": {
                            "retrieved_docs_count": agent_result['retrieved_docs_count'],
                            "retry_count": agent_result['retry_count'],
                            "relevance_score": agent_result['relevance_score']
                        }
                    }
                    
                    print(f"\n✅ Full Pipeline Complete!")
                    print("📦 Final response structure ready for frontend")
                    print(f"🔊 Audio data size: {len(audio_result.get('audio_base64', ''))} characters")
                    
                else:
                    print("⚠️ Speech generation failed")
            
        except Exception as e:
            print(f"❌ Full pipeline error: {e}")
    
    async def test_sentiment_analysis(self):
        """Test sentiment analysis with different types of queries."""
        print("\n😊 Testing Sentiment Analysis...")
        
        sentiment_test_cases = [
            ("I'm confused about plant care", "confused"),
            ("This is so exciting! Tell me more!", "excited"), 
            ("What is photosynthesis?", "curious"),
            ("I can't get this to work, help!", "frustrated"),
            ("How do I water plants?", "neutral")
        ]
        
        for test_input, expected in sentiment_test_cases:
            try:
                detected = self.agent.sentiment_analyzer.analyze(test_input)
                status = "✅" if detected == expected else "⚠️"
                print(f"{status} '{test_input}' → {detected} (expected: {expected})")
            except Exception as e:
                print(f"❌ Error analyzing '{test_input}': {e}")
    
    async def run_all_tests(self):
        """Run all integration tests."""
        print("🧪 RAG Agent Integration Tests")
        print("="*50)
        
        # Setup
        if not await self.setup_services():
            print("❌ Service setup failed, aborting tests")
            return
        
        # Run tests
        await self.test_sentiment_analysis()
        await self.test_basic_rag()
        await self.test_self_correction()
        await self.test_tts_pipeline()
        
        # Interactive full pipeline test
        print("\n" + "="*50)
        full_test = input("Run full pipeline test? (y/n): ").strip().lower()
        if full_test == 'y':
            await self.test_full_pipeline()
        
        print(f"\n🎉 Integration testing complete!")
        print("Next steps:")
        print("1. Start the API: python -m app.main")
        print("2. Test endpoints with curl or Postman")
        print("3. Build frontend integration")

async def main():
    """Main test execution."""
    tester = IntegrationTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())