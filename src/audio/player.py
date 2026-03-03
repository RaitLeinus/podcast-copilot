"""
speaker.py - Text-to-speech playback.

speak(text)           — OpenAI TTS with macOS say fallback
speak_stream(iter)    — Streams PCM16 audio chunks (24kHz mono) from gpt-4o-audio-preview
"""

import os
import subprocess
import tempfile

import numpy as np


def speak(text: str):
    """Speak text using OpenAI TTS (tts-1), falling back to macOS say."""
    print(f"\n💬 EXPLANATION:\n{text}\n")
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.audio.speech.create(
                model="tts-1",
                voice="nova",
                input=text
            )
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp_path = f.name
                f.write(response.content)
            subprocess.run(["afplay", tmp_path], check=True)
            os.unlink(tmp_path)
            return
        except Exception as e:
            print(f"OpenAI TTS error: {e}, falling back to say")
    try:
        subprocess.run(["say", "-v", "Daniel", "-r", "185", text], check=True)
    except Exception as e:
        print(f"TTS error: {e}")


def speak_stream(chunk_iter):
    """Play streaming PCM16 audio chunks (24kHz mono) from gpt-4o-audio-preview."""
    import sounddevice as sd
    stream = sd.OutputStream(samplerate=24000, channels=1, dtype="int16")
    stream.start()
    try:
        for pcm_bytes in chunk_iter:
            if pcm_bytes:
                stream.write(np.frombuffer(pcm_bytes, dtype=np.int16))
    finally:
        stream.stop()
        stream.close()
