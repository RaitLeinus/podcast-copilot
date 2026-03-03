"""
explainer.py - Uses GPT-4o to explain podcast content based on recent transcript context.
"""

import base64
import os
from openai import OpenAI


SYSTEM_PROMPT = """You are a podcast listening assistant. Your response will be read aloud via text-to-speech.
Rules:
- 1-2 sentences max, no exceptions
- No markdown, bullet points, or special characters
- Speak naturally and directly, no filler like "Sure!" or "Great question!"
- No intro like "Based on the transcript, here's an explanation:" — just jump straight to the explanation
- Give background about things/people/concepts mentioned in the transcript but dont repeat the transcript text
- If the user asked about a specific topic, focus on that in your explanation
"""


class Explainer:
    def __init__(self):
        self._client = None

    @property
    def client(self):
        if self._client is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set.")
            self._client = OpenAI(api_key=api_key)
        return self._client

    def explain_audio_stream(self, transcript_context: str, focus: str = None):
        """
        Stream a spoken explanation as raw PCM16 audio chunks (24kHz, mono).
        Yields bytes objects. Uses gpt-4o-audio-preview to combine generation + TTS in one pass.
        """
        if not transcript_context.strip():
            return

        user_message = f"Recent podcast transcript:\n---\n{transcript_context}\n---\n\n"
        if focus:
            user_message += f'The listener asked: "{focus}"\n'

        response = self.client.chat.completions.create(
            model="gpt-4o-audio-preview",
            modalities=["text", "audio"],
            audio={"voice": "nova", "format": "pcm16"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            stream=True,
        )
        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.model_dump()
            audio = delta.get("audio") or {}
            if audio.get("data"):
                yield base64.b64decode(audio["data"])
