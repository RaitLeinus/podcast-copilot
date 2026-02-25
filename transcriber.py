"""
transcriber.py - Transcribes audio chunks using OpenAI Whisper API.

Uses the fast whisper-1 model via API. For fully offline/private use,
swap with local faster-whisper: https://github.com/SYSTRAN/faster-whisper
"""

import io
import os
import wave
import tempfile
import numpy as np
from openai import OpenAI


class Transcriber:
    def __init__(self, model="whisper-1", language=None):
        """
        model: "whisper-1" for OpenAI API
        language: Optional ISO language code (e.g. "en", "et"). None = auto-detect.
        """
        self.model = model
        self.language = language
        self._client = None

    @property
    def client(self):
        if self._client is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set.")
            self._client = OpenAI(api_key=api_key)
        return self._client

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe a numpy float32 audio array (mono, 16kHz).
        Returns transcribed text or empty string.
        """
        if audio is None or len(audio) == 0:
            return ""

        # Convert float32 [-1, 1] to int16
        audio_int16 = (audio * 32767).astype(np.int16)

        # Write to in-memory WAV
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(16000)
            wf.writeframes(audio_int16.tobytes())
        wav_buffer.seek(0)
        wav_buffer.name = "audio.wav"  # OpenAI requires a filename

        try:
            kwargs = {
                "model": self.model,
                "file": wav_buffer,
                "response_format": "text",
            }
            if self.language:
                kwargs["language"] = self.language

            response = self.client.audio.transcriptions.create(**kwargs)
            return response if isinstance(response, str) else str(response)

        except Exception as e:
            print(f"Transcription API error: {e}")
            return ""


class LocalTranscriber:
    """
    Alternative: fully local transcription using faster-whisper.
    No API key required. Install: pip install faster-whisper

    Usage: replace Transcriber() with LocalTranscriber() in app.py
    """
    def __init__(self, model_size="base.en", device="cpu"):
        try:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(model_size, device=device, compute_type="int8")
            print(f"✓ Local Whisper model loaded: {model_size}")
        except ImportError:
            raise ImportError(
                "faster-whisper not installed. Run: pip install faster-whisper"
            )

    def transcribe(self, audio: np.ndarray) -> str:
        segments, info = self.model.transcribe(audio, beam_size=5)
        return " ".join(segment.text for segment in segments).strip()
