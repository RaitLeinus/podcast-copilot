"""
player.py - Text-to-speech playback.

speak(text)           — OpenAI TTS with macOS say fallback
speak_stream(iter)    — Streams PCM16 audio chunks (24kHz mono) from gpt-4o-audio-preview
"""

import io
import os
import subprocess
import tempfile
import wave


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
    """Play streaming PCM16 audio chunks (24kHz mono) via afplay.

    Collects all chunks into a WAV file, then plays via macOS afplay
    which automatically uses the current system default output device.
    """
    chunks = []
    for pcm_bytes in chunk_iter:
        if pcm_bytes:
            chunks.append(pcm_bytes)
    if not chunks:
        return
    pcm_data = b"".join(chunks)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
        with wave.open(f, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(pcm_data)
    try:
        subprocess.run(["afplay", tmp_path], check=True)
    finally:
        os.unlink(tmp_path)
