import base64
import tempfile
import os
import json
import logging
import requests
from typing import Dict, List, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(__name__)

class TTSService:
    def __init__(self, voice_id: Optional[str] = None):
        """
        Initialize TTS service with ElevenLabs integration.
        
        Args:
            voice_id: ElevenLabs voice ID (defaults to a friendly voice)
        """
        self.voice_id = voice_id or "21m00Tcm4TlvDq8ikWAM"
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        self.base_url = "https://api.elevenlabs.io/v1"
        
        self.temp_dir = Path(tempfile.gettempdir()) / "rag_agent_audio"
        self.temp_dir.mkdir(exist_ok=True)
        
        self.voice_settings = {
            "stability": 0.5,        
            "similarity_boost": 0.75,
            "style": 0.3,          
            "use_speaker_boost": True 
        }
    
    async def generate_speech(self, text: str, voice_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate speech audio from text using ElevenLabs.
        
        Args:
            text: Text to convert to speech
            voice_id: Optional voice override
            
        Returns:
            Dict containing audio_base64, audio_path, and metadata
        """
        try:
            logger.info(f"🎵 Generating TTS audio with ElevenLabs...")
            
            # Use provided voice or default
            voice = voice_id or self.voice_id
            audio_path = self.temp_dir / f"tts_{hash(text)}.mp3"
            
            if self.api_key:
                return await self._generate_elevenlabs_audio(text, voice, audio_path)
            else:
                logger.warning("No ElevenLabs API key found, using demo mode")
                return await self._generate_demo_audio(text, audio_path)
            
        except Exception as e:
            logger.error(f"TTS generation error: {e}")
            return {
                "audio_base64": "",
                "audio_path": "",
                "duration": 0,
                "success": False,
                "error": str(e)
            }
    
    async def _generate_elevenlabs_audio(self, text: str, voice_id: str, audio_path: Path) -> Dict[str, Any]:
        """Generate audio using ElevenLabs API."""
        try:
            url = f"{self.base_url}/text-to-speech/{voice_id}"
            
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.api_key
            }
            
            data = {
                "text": text,
                "model_id": "eleven_monolingual_v1",  # Fast, good quality model
                "voice_settings": self.voice_settings
            }
            
            logger.info(f"Calling ElevenLabs API for voice {voice_id}...")
            response = requests.post(url, json=data, headers=headers, timeout=30)
            
            if response.status_code == 200:
                # Save audio file
                with open(audio_path, 'wb') as f:
                    f.write(response.content)
                
                # Convert to base64 for API response
                audio_base64 = base64.b64encode(response.content).decode('utf-8')
                
                # Estimate duration (rough calculation)
                duration = self._estimate_audio_duration(text)
                
                logger.info(f"✅ ElevenLabs audio generated successfully")
                return {
                    "audio_base64": audio_base64,
                    "audio_path": str(audio_path),
                    "duration": duration,
                    "success": True,
                    "voice_used": voice_id,
                    "text": text,
                    "file_size": len(response.content),
                    "method": "elevenlabs"
                }
            
            elif response.status_code == 401:
                logger.error("ElevenLabs API: Invalid API key")
                raise Exception("Invalid ElevenLabs API key")
            elif response.status_code == 429:
                logger.error("ElevenLabs API: Rate limit exceeded")
                raise Exception("ElevenLabs rate limit exceeded")
            else:
                logger.error(f"ElevenLabs API error: {response.status_code} - {response.text}")
                raise Exception(f"ElevenLabs API error: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error calling ElevenLabs: {e}")
            raise Exception(f"Network error: {e}")
    
    def _estimate_audio_duration(self, text: str) -> float:
        """Estimate audio duration based on text length."""
        # Average speaking rate: ~150 words per minute
        # Average word length: ~5 characters
        words = len(text) / 5  # Rough word count
        duration = (words / 150) * 60  # Convert to seconds
        return max(1.0, duration)  # Minimum 1 second
    
    async def generate_lip_sync(self, audio_path: str) -> Dict[str, Any]:
        """
        Generate lip-sync data from audio file.
        
        For now, generates demo data. In production, you could integrate:
        - Rhubarb Lip Sync
        - ElevenLabs lip-sync API (if available)
        - Custom phoneme detection
        """
        try:
            logger.info("👄 Generating lip-sync data...")
            
            # Get audio duration from file if possible
            duration = self._get_audio_duration(audio_path)
            
            # Generate more realistic lip-sync based on duration
            return await self._generate_realistic_lip_sync(audio_path, duration)
                
        except Exception as e:
            logger.error(f"Lip-sync generation error: {e}")
            return {
                "visemes": [],
                "duration": 0,
                "success": False,
                "error": str(e)
            }
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """Get actual audio duration from file."""
        try:
            # Try to get duration from file size (rough estimate for MP3)
            if os.path.exists(audio_path):
                file_size = os.path.getsize(audio_path)
                # Rough estimate: ~1KB per 0.1 seconds for speech MP3
                duration = file_size / 10000
                return max(1.0, duration)
        except Exception:
            pass
        
        return 2.0  # Default fallback
    
    async def _generate_realistic_lip_sync(self, audio_path: str, duration: float) -> Dict[str, Any]:
        """Generate more realistic lip-sync data based on audio duration."""
        
        # Common viseme patterns for natural speech
        viseme_patterns = [
            "X",  # Silence/rest
            "A",  # Open vowels (ah, aa)
            "B",  # Closed consonants (p, b, m)
            "C",  # Front vowels (e, i)  
            "D",  # Tongue consonants (t, d, s, z)
            "E",  # R sounds
            "F",  # Lip consonants (f, v)
            "G",  # Throat consonants (k, g)
            "H",  # Back vowels (o, u)
        ]
        
        # Generate visemes at regular intervals
        visemes = []
        num_frames = int(duration * 8)  # 8 frames per second
        
        for i in range(num_frames):
            time_point = (i / 8)  # Convert to seconds
            
            # Start and end with silence
            if time_point < 0.1 or time_point > duration - 0.1:
                viseme = "X"
            else:
                # Simulate natural speech patterns
                # More vowels and common consonants
                import random
                random.seed(int(time_point * 100))  # Deterministic but varied
                
                weights = [0.1, 0.25, 0.15, 0.2, 0.15, 0.05, 0.05, 0.03, 0.02]
                viseme = random.choices(viseme_patterns, weights=weights)[0]
            
            visemes.append({
                "time": round(time_point, 2),
                "value": viseme
            })
        
        return {
            "visemes": visemes,
            "duration": duration,
            "success": True,
            "method": "realistic_simulation",
            "frame_count": len(visemes)
        }
    
    async def _generate_demo_audio(self, text: str, audio_path: Path) -> Dict[str, Any]:
        """Generate demo audio when ElevenLabs is not available."""
        logger.info("Using demo mode (no ElevenLabs API key)")
        
        duration = self._estimate_audio_duration(text)
        
        return {
            "audio_base64": self._create_demo_audio_base64(),
            "audio_path": str(audio_path),
            "duration": duration,
            "success": True,
            "voice_used": "demo",
            "text": text,
            "method": "demo"
        }
    
    def _create_demo_audio_base64(self) -> str:
        """Create a minimal demo audio file as base64."""
        # Create a tiny silent WAV file for demo purposes
        wav_header = b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00'
        return base64.b64encode(wav_header).decode('utf-8')
    
    async def get_available_voices(self) -> Dict[str, Any]:
        """Get list of available voices from ElevenLabs."""
        if not self.api_key:
            return {
                "error": "No API key provided",
                "demo_voices": [
                    {"voice_id": "demo", "name": "Demo Voice", "category": "demo"}
                ]
            }
        
        try:
            url = f"{self.base_url}/voices"
            headers = {"xi-api-key": self.api_key}
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                voices_data = response.json()
                return {
                    "success": True,
                    "voices": [
                        {
                            "voice_id": voice["voice_id"],
                            "name": voice["name"],
                            "category": voice.get("category", "custom"),
                            "description": voice.get("description", ""),
                            "preview_url": voice.get("preview_url")
                        }
                        for voice in voices_data.get("voices", [])
                    ]
                }
            else:
                return {"error": f"API error: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Error getting voices: {e}")
            return {"error": str(e)}
    
    async def test_tts_pipeline(self, text: str = "Hello! I'm your AI tutor, and I'm excited to help you learn about plants today.") -> Dict[str, Any]:
        """Test the complete TTS pipeline with ElevenLabs."""
        try:
            logger.info("Testing ElevenLabs TTS pipeline...")
            
            # Test audio generation
            audio_result = await self.generate_speech(text)
            
            if audio_result.get("success"):
                # Test lip-sync generation
                lip_sync_result = await self.generate_lip_sync(audio_result["audio_path"])
                
                return {
                    "pipeline_success": True,
                    "audio_test": {
                        "success": audio_result["success"],
                        "method": audio_result.get("method"),
                        "duration": audio_result.get("duration"),
                        "voice_used": audio_result.get("voice_used"),
                        "file_size": audio_result.get("file_size", 0)
                    },
                    "lip_sync_test": {
                        "success": lip_sync_result["success"],
                        "method": lip_sync_result.get("method"),
                        "frame_count": len(lip_sync_result.get("visemes", []))
                    },
                    "test_text": text,
                    "api_status": "elevenlabs" if self.api_key else "demo"
                }
            else:
                return {
                    "pipeline_success": False,
                    "audio_test": audio_result,
                    "lip_sync_test": {"success": False, "error": "Audio generation failed"},
                    "test_text": text
                }
                
        except Exception as e:
            logger.error(f"TTS pipeline test failed: {e}")
            return {
                "pipeline_success": False,
                "error": str(e),
                "test_text": text
            }
    
    def get_voice_recommendations(self) -> Dict[str, str]:
        """Get recommended voice IDs for different use cases."""
        return {
            # These are popular ElevenLabs voices - you may need to check availability
            "friendly_female": "21m00Tcm4TlvDq8ikWAM",  # Rachel
            "professional_male": "29vD33N1CtxCmqQRPOHJ",   # Drew  
            "warm_female": "pNInz6obpgDQGcFmaJgB",        # Adam
            "energetic_female": "EXAVITQu4vr4xnSDxMaL",   # Bella
            "calm_male": "VR6AewLTigWG4xSOukaG",          # Arnold
            
            # Note: These voice IDs might change. Check your ElevenLabs dashboard
            # for current voice IDs, or use the get_available_voices() method
        }

# Example usage and voice testing
if __name__ == "__main__":
    import asyncio
    
    async def test_elevenlabs():
        tts = TTSService()
        
        print("🔊 Testing ElevenLabs TTS Service...")
        
        # Test basic functionality
        result = await tts.test_tts_pipeline()
        print("Pipeline Test Results:")
        print(json.dumps(result, indent=2))
        
        # Test available voices
        print("\n🎤 Getting available voices...")
        voices = await tts.get_available_voices()
        if voices.get("success"):
            print(f"Found {len(voices['voices'])} available voices:")
            for voice in voices['voices'][:5]:  # Show first 5
                print(f"  - {voice['name']} ({voice['voice_id']})")
        else:
            print(f"Voice lookup failed: {voices.get('error')}")
        
        print(f"\n💡 Recommended voices:")
        recommendations = tts.get_voice_recommendations()
        for use_case, voice_id in recommendations.items():
            print(f"  {use_case}: {voice_id}")
    
    asyncio.run(test_elevenlabs())