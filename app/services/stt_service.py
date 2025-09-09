import whisper
import sounddevice as sd
import numpy as np
import queue
import threading

class STTService:
    def __init__(self, model_size="base"):
        """Initialize the STT service with a Whisper model."""
        self.model = whisper.load_model(model_size)
        self.audio_queue = queue.Queue()
        self.sample_rate = 16000
        self.is_recording = False

    def callback(self, indata, frames, time, status):
        """Callback function to handle audio input from the microphone."""
        if status:
            print(status)
        self.audio_queue.put(indata.copy())

    def record_audio(self):
        """Record audio from the microphone in a separate thread."""
        with sd.InputStream(samplerate=self.sample_rate, channels=1, callback=self.callback,
                           dtype='float32', blocksize=1600):
            print("Recording... Press Ctrl+C to stop or wait 5 seconds.")
            while self.is_recording:
                sd.sleep(100)

    def transcribe(self, duration=5):
        """Transcribe audio input for a specified duration."""
        self.is_recording = True
        audio_data = []

        # Start recording in a separate thread
        record_thread = threading.Thread(target=self.record_thread)
        record_thread.start()

        sd.sleep(int(duration * 1000)) 
        self.is_recording = False
        record_thread.join()

        while not self.audio_queue.empty():
            audio_data.append(self.audio_queue.get())

        if audio_data:
            audio_array = np.concatenate(audio_data, axis=0)
            result = self.model.transcribe(audio_array, fp16=False)
            return result["text"]
        return ""

    def record_thread(self):
        """Wrapper for record_audio to handle thread execution."""
        self.record_audio()

if __name__ == "__main__":
    stt = STTService()
    transcription = stt.transcribe(duration=5)
    print(f"Transcribed text: {transcription}")