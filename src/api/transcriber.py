"""
transcriber.py - Transcribes audio chunks using OpenAI Whisper API.
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

    def transcribe(self, audio: np.ndarray, prompt: str = None, language: str = None) -> str:
        """
        Transcribe a numpy float32 audio array (mono, 16kHz).
        Returns transcribed text or empty string.
        prompt: optional hint to guide Whisper's language model (improves short-command accuracy).
        language: override instance language (e.g. "en") — important for short clips where
                  auto-detection is unreliable.
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
            lang = language or self.language
            if lang:
                kwargs["language"] = lang
            if prompt:
                kwargs["prompt"] = prompt

            response = self.client.audio.transcriptions.create(**kwargs)
            return response if isinstance(response, str) else str(response)

        except Exception as e:
            print(f"Transcription API error: {e}")
            return ""
