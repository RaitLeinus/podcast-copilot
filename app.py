"""
Podcast Copilot - Mac Menu Bar App
Architecture:
  - System audio is buffered locally in RAM (never sent anywhere)
  - Microphone uses on-device Porcupine wake word detection (no API)
  - Only when you say "explain that": buffered audio → Whisper → GPT-4o
"""

import threading
import time
import os
import subprocess
import concurrent.futures
from collections import deque

import rumps
import numpy as np

from audio_buffer import RollingAudioBuffer
from audio_capture import AudioCapture
from wake_word import PorcupineWakeWordDetector, FallbackWakeWordDetector
from transcriber import Transcriber
from explainer import Explainer

SAMPLE_RATE = 16000
BUFFER_DURATION_SECONDS = 30
CAPTURE_USER_COMMAND_DURATION = 3.0



class PodcastCopilot(rumps.App):
    def __init__(self):
        super().__init__(
            "🎙",
            quit_button=rumps.MenuItem("Quit Podcast Copilot", key="q")
        )

        self.menu = [
            rumps.MenuItem("Status: Idle", callback=None),
            rumps.MenuItem("Buffer: 0s captured", callback=None),
            rumps.separator,
            rumps.MenuItem("▶ Start Listening", callback=self.toggle_listening),
            rumps.MenuItem("🔊 Test Explain", callback=self.test_explain),
            rumps.separator,
            rumps.MenuItem("⚙ Settings", callback=self.open_settings),
        ]

        self.status_item = self.menu["Status: Idle"]
        self.buffer_item = self.menu["Buffer: 0s captured"]
        self.toggle_item = self.menu["▶ Start Listening"]

        self.is_listening = False
        self.is_explaining = False

        # THE KEY CHANGE: raw audio buffer in RAM, nothing goes to API until triggered
        self.audio_buffer = RollingAudioBuffer(
            sample_rate=SAMPLE_RATE,
            max_seconds=BUFFER_DURATION_SECONDS
        )

        self.audio_capture = AudioCapture(sample_rate=SAMPLE_RATE)
        self.transcriber = Transcriber()
        self.explainer = Explainer()
        self.wake_detector = None  # initialized on start

        # Update buffer display every 5 seconds
        self._buffer_timer = rumps.Timer(self._update_buffer_display, 5)
        self._buffer_timer.start()

    def set_status(self, text, icon="🎙"):
        self.title = icon
        self.status_item.title = f"Status: {text}"

    def _update_buffer_display(self, _=None):
        seconds = self.audio_buffer.duration_seconds
        if seconds < 60:
            label = f"Buffer: {int(seconds)}s captured"
        else:
            label = f"Buffer: {int(seconds) // 60}m {int(seconds) % 60}s captured"
        self.buffer_item.title = label

    @rumps.clicked("▶ Start Listening")
    def toggle_listening(self, sender):
        if not self.is_listening:
            self.start_listening()
        else:
            self.stop_listening()

    def start_listening(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            rumps.alert(
                title="API Key Missing",
                message=(
                    "Please set OPENAI_API_KEY in your environment.\n\n"
                    "Use ⚙ Settings to enter it, or:\n"
                    "  export OPENAI_API_KEY=sk-...\n\n"
                    "Note: Your API key is only used when you trigger an explanation.\n"
                    "Audio is stored locally until then."
                ),
                ok="OK"
            )
            return

        # Choose wake word detector based on what's configured
        porcupine_key = os.environ.get("PORCUPINE_ACCESS_KEY")
        _bundled_model = os.path.join(os.path.dirname(os.path.abspath(__file__)), "explain_en_mac_v4_0_0.ppn")
        porcupine_model = os.environ.get("PORCUPINE_MODEL_PATH") or (
            _bundled_model if os.path.exists(_bundled_model) else None
        )

        if porcupine_key and porcupine_model:
            print("✓ Using Porcupine wake word detector (on-device, fast)")
            self.wake_detector = PorcupineWakeWordDetector(
                access_key=porcupine_key,
                keyword_path=porcupine_model,
                callback=self.on_wake_word
            )
        else:
            print("⚠ Porcupine not configured — using energy-based fallback detector.")
            print("  For better accuracy, see README for Porcupine setup.")
            self.wake_detector = FallbackWakeWordDetector(callback=self.on_wake_word)

        self.is_listening = True
        self.toggle_item.title = "⏹ Stop Listening"
        self.set_status("Buffering locally...", "🔴")

        # Start capturing system audio into local RAM buffer — no API calls here
        self.audio_capture.start(callback=self.audio_buffer.append)

        # Start mic-based wake word detection — fully on-device
        self.wake_detector.start()

        rumps.notification(
            title="Podcast Copilot",
            subtitle="Listening — audio stored locally only",
            message="Say 'explain that' to explain what you just heard."
        )

    def stop_listening(self):
        self.is_listening = False
        self.toggle_item.title = "▶ Start Listening"
        self.set_status("Idle", "🎙")
        self.buffer_item.title = "Buffer: 0s captured"

        self.audio_capture.stop()
        if self.wake_detector:
            self.wake_detector.stop()
            self.wake_detector = None

        self.audio_buffer.clear()

    def on_wake_word(self):
        """Called from wake word detector thread when trigger phrase is heard."""
        if self.is_explaining:
            print("Already explaining, ignoring wake word.")
            return

        print("Wake word detected — triggering explanation")
        threading.Thread(target=self._do_explain, daemon=True).start()

    def _capture_user_command(self, max_duration=4.0):
        """Record mic with VAD:
        - Stops ~0.6s after speech ends
        - Stops after 1.5s if no speech detected at all (user said nothing)
        - Hard cap at max_duration
        """
        import sounddevice as sd
        chunk_seconds = 0.05  # 50ms chunks
        chunk_size = int(SAMPLE_RATE * chunk_seconds)
        silence_threshold = 0.015
        silence_needed = int(0.6 / chunk_seconds)   # 0.6s post-speech silence → stop
        pre_speech_timeout = int(0.8 / chunk_seconds)  # 0.8s with no speech → give up
        max_chunks = int(max_duration / chunk_seconds)

        speech_confirm_needed = int(0.15 / chunk_seconds)  # 150ms sustained to confirm speech

        recorded = []
        speech_started = False
        speech_confirm_count = 0
        silence_count = 0
        pre_speech_count = 0

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=chunk_size) as stream:
            for _ in range(max_chunks):
                chunk, _ = stream.read(chunk_size)
                chunk = chunk.flatten()
                rms = float(np.sqrt(np.mean(chunk ** 2)))
                recorded.append(chunk)

                if rms > silence_threshold:
                    silence_count = 0
                    pre_speech_count = 0
                    if not speech_started:
                        speech_confirm_count += 1
                        if speech_confirm_count >= speech_confirm_needed:
                            speech_started = True
                elif speech_started:
                    speech_confirm_count = 0
                    silence_count += 1
                    if silence_count >= silence_needed:
                        print(f"VAD: end of speech, stopped after {len(recorded) * chunk_seconds:.1f}s")
                        break
                else:
                    speech_confirm_count = 0
                    pre_speech_count += 1
                    if pre_speech_count >= pre_speech_timeout:
                        print("VAD: no speech detected, stopping early")
                        break

        return np.concatenate(recorded) if recorded else np.zeros(0, dtype=np.float32)

    def _do_explain(self):
        """
        Core flow:
        1. Listen for user's follow-up command (e.g., "explain nazis")
        2. Pause media
        3. Snapshot the local audio buffer
        4. Send to Whisper (first + only API call until this moment)
        5. Send transcript + optional focus topic to GPT-4o
        6. Speak explanation
        7. Resume media
        """
        self.is_explaining = True

        # Pause media and snapshot buffer immediately — buffer is system audio only,
        # safe to snapshot before mic recording starts
        self._control_media("pause")
        audio_snapshot = self.audio_buffer.get_audio()
        buffer_seconds = len(audio_snapshot) / SAMPLE_RATE
        print(f"Audio snapshot: {buffer_seconds:.1f}s")

        if buffer_seconds < 5:
            self._speak("I don't have enough audio context yet. Try again after listening for a bit.")
            self._control_media("play")
            self.is_explaining = False
            self.set_status("Buffering locally...", "🔴")
            return

        # Start Whisper on the buffer immediately — runs during command recording
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        print(f"Submitting {buffer_seconds:.0f}s buffer to Whisper...")
        future_transcript = pool.submit(self.transcriber.transcribe, audio_snapshot)

        # Prime the capture queue before the chime so no audio is missed
        if self.wake_detector and hasattr(self.wake_detector, "start_capture"):
            self.wake_detector.start_capture()

        self.set_status("Listening...", "👂")
        subprocess.Popen(["afplay", "/System/Library/Sounds/Tink.aiff"])

        if self.wake_detector and hasattr(self.wake_detector, "get_capture_audio"):
            user_command_audio = self.wake_detector.get_capture_audio(
                max_duration=CAPTURE_USER_COMMAND_DURATION
            )
        else:
            user_command_audio = self._capture_user_command(max_duration=CAPTURE_USER_COMMAND_DURATION)

        self.set_status("Transcribing...", "⏳")

        # Submit command transcription — buffer Whisper may already be done by now
        future_command = pool.submit(
            self.transcriber.transcribe,
            user_command_audio,
            "explain, what is, tell me about, who is, why did, how does",
            "en"
        )
        pool.shutdown(wait=True)
        

        try:
            transcript = future_transcript.result()
            print(f"Transcript:\n{transcript}")
        except Exception as e:
            print(f"Transcription error: {e}")
            self._speak("Sorry, transcription failed. Check your API key and internet connection.")
            self._control_media("play")
            self.is_explaining = False
            self.set_status("Buffering locally...", "🔴")
            return

        if not transcript.strip():
            self._speak("Couldn't make out what was being said. Check that BlackHole is set up correctly.")
            self._control_media("play")
            self.is_explaining = False
            self.set_status("Buffering locally...", "🔴")
            return

        focus = None
        try:
            user_command = future_command.result().strip()
            print(f"User command: {user_command!r}")
            if user_command:
                focus = user_command
        except Exception:
            pass  # no focus — explain the most recent topic

        # --- SECOND API CALL: GPT-4o audio stream (generation + TTS in one pass) ---
        self.set_status("Explaining...", "💬")
        try:
            self._speak_stream(self.explainer.explain_audio_stream(transcript, focus=focus))
        except Exception as e:
            print(f"Explanation error: {e}")
            self._speak("Sorry, I had trouble generating an explanation.")
            self._control_media("play")
            self.is_explaining = False
            self.set_status("Buffering locally...", "🔴")
            return

        # Resume playback
        time.sleep(0.3)
        self._control_media("play")

        self.is_explaining = False
        self.set_status("Buffering locally...", "🔴")

    def _control_media(self, action):
        """Pause or resume media via F8 media key (works for Spotify, YouTube, Chrome, etc.)"""
        script = 'tell application "System Events" to key code 16'
        try:
            subprocess.run(["osascript", "-e", script], check=True, capture_output=True)
        except Exception:
            pass  # No Accessibility permission — Spotify AppleScript fallback below
            applescript = (
                'tell application "Spotify" to pause'
                if action == "pause"
                else 'tell application "Spotify" to play'
            )
            try:
                subprocess.run(["osascript", "-e", applescript], capture_output=True)
            except Exception:
                pass

    def _speak(self, text):
        """Speak text using OpenAI TTS (tts-1), falling back to macOS say."""
        import tempfile
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

    def _speak_stream(self, chunk_iter):
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

    @rumps.clicked("🔊 Test Explain")
    def test_explain(self, sender):
        threading.Thread(target=self._test_explain_worker, daemon=True).start()

    def _test_explain_worker(self):
        self.set_status("Testing...", "💬")
        test_transcript = (
            "Today we're going to talk about transformer neural networks. "
            "The key innovation is the attention mechanism, specifically self-attention, "
            "which lets the model look at all positions in the input sequence simultaneously "
            "rather than processing tokens one by one like older RNNs did. "
            "This is what makes models like GPT and BERT so powerful. "
            "The query, key, and value matrices are learned during training and allow "
            "the model to figure out which parts of the input are most relevant "
            "when generating each output token."
        )
        try:
            explanation = self.explainer.explain(test_transcript)
            self._speak(explanation)
        except Exception as e:
            rumps.alert("Test Error", str(e))
        self.set_status(
            "Idle" if not self.is_listening else "Buffering locally...",
            "🎙" if not self.is_listening else "🔴"
        )

    @rumps.clicked("⚙ Settings")
    def open_settings(self, sender):
        response = rumps.Window(
            message="OpenAI API Key (only used when you trigger an explanation):",
            title="Podcast Copilot — Settings",
            default_text=os.environ.get("OPENAI_API_KEY", ""),
            ok="Save",
            cancel="Cancel",
            dimensions=(420, 30)
        ).run()
        if response.clicked and response.text:
            os.environ["OPENAI_API_KEY"] = response.text
            _save_env("OPENAI_API_KEY", response.text)
            rumps.notification("Saved", "OpenAI API key saved", "")

        response2 = rumps.Window(
            message="Porcupine Access Key (optional — for faster on-device wake word):",
            title="Podcast Copilot — Settings",
            default_text=os.environ.get("PORCUPINE_ACCESS_KEY", ""),
            ok="Save",
            cancel="Skip",
            dimensions=(420, 30)
        ).run()
        if response2.clicked and response2.text:
            os.environ["PORCUPINE_ACCESS_KEY"] = response2.text
            _save_env("PORCUPINE_ACCESS_KEY", response2.text)

        response3 = rumps.Window(
            message="Porcupine model path (.ppn file — leave blank if not using Porcupine):",
            title="Podcast Copilot — Settings",
            default_text=os.environ.get("PORCUPINE_MODEL_PATH", ""),
            ok="Save",
            cancel="Skip",
            dimensions=(420, 30)
        ).run()
        if response3.clicked and response3.text:
            os.environ["PORCUPINE_MODEL_PATH"] = response3.text
            _save_env("PORCUPINE_MODEL_PATH", response3.text)


def _save_env(key, value):
    """Persist key=value to ~/.podcast_copilot_env"""
    env_file = os.path.expanduser("~/.podcast_copilot_env")
    lines = []
    found = False
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                if line.startswith(f"{key}="):
                    lines.append(f"{key}={value}\n")
                    found = True
                else:
                    lines.append(line)
    if not found:
        lines.append(f"{key}={value}\n")
    with open(env_file, "w") as f:
        f.writelines(lines)


def load_env():
    env_file = os.path.expanduser("~/.podcast_copilot_env")
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip())


if __name__ == "__main__":
    load_env()
    app = PodcastCopilot()
    app.run()
