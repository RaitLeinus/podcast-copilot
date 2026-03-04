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

import numpy as np
import sounddevice as sd

import rumps

from audio import RollingAudioBuffer, AudioCapture, find_input_device, list_input_devices, speak, speak_stream
from wakeword import PorcupineWakeWordDetector, FallbackWakeWordDetector
from api import Transcriber, Explainer
from util import control_media, save_env, load_env

SAMPLE_RATE = 16000
BUFFER_DURATION_SECONDS = 30
CAPTURE_USER_COMMAND_DURATION = 3.0


class PodcastCopilot(rumps.App):
    def __init__(self):
        super().__init__(
            "🎙",
            quit_button=rumps.MenuItem("Quit Podcast Copilot", key="q")
        )

        self.mic_menu = rumps.MenuItem("🎤 Microphone")
        self._populate_mic_menu()

        self.menu = [
            rumps.MenuItem("Status: Idle", callback=None),
            rumps.MenuItem("Buffer: 0s captured", callback=None),
            rumps.separator,
            rumps.MenuItem("▶ Start Listening", callback=self.toggle_listening),
            rumps.separator,
            self.mic_menu,
            rumps.MenuItem("⚙ Settings", callback=self.open_settings),
        ]

        self.status_item = self.menu["Status: Idle"]
        self.buffer_item = self.menu["Buffer: 0s captured"]
        self.toggle_item = self.menu["▶ Start Listening"]

        self.is_listening = False
        self.is_explaining = False

        self.audio_buffer = RollingAudioBuffer(
            sample_rate=SAMPLE_RATE,
            max_seconds=BUFFER_DURATION_SECONDS
        )

        self.audio_capture = AudioCapture(sample_rate=SAMPLE_RATE)
        self.transcriber = Transcriber()
        self.explainer = Explainer()
        self.wake_detector = None  # initialized on start

        # Track default input device to detect changes
        self._last_default_input = sd.default.device[0]

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

        # Refresh devices: stop stream, re-enumerate PortAudio, restart stream
        # PortAudio caches the device list — re-init is the only way to refresh
        if self.is_listening and self.wake_detector:
            try:
                self.wake_detector.stop_stream()
                sd._terminate()
                sd._initialize()
            except Exception:
                pass

            # Refresh mic menu if device list changed
            try:
                current_devices = frozenset(name for _, name in list_input_devices())
                if not hasattr(self, '_last_known_devices') or current_devices != self._last_known_devices:
                    self._last_known_devices = current_devices
                    self._populate_mic_menu()
            except Exception:
                pass

            # Re-resolve device after re-enumeration (indices change)
            mic_name = os.environ.get("MIC_DEVICE", "")
            if mic_name:
                mic_device, _ = find_input_device(mic_name)
                self.wake_detector.device = mic_device
            else:
                self.wake_detector.device = None

            # Restart the stream
            try:
                self.wake_detector.start_stream()
            except Exception as e:
                print(f"Failed to restart mic stream: {e}")
                self.set_status("Mic error — try Stop/Start", "⚠️")
        else:
            # Not listening — just refresh the menu
            try:
                sd._terminate()
                sd._initialize()
                current_devices = frozenset(name for _, name in list_input_devices())
                if not hasattr(self, '_last_known_devices') or current_devices != self._last_known_devices:
                    self._last_known_devices = current_devices
                    self._populate_mic_menu()
            except Exception:
                pass

    def _populate_mic_menu(self):
        if self.mic_menu._menu is not None:
            self.mic_menu.clear()
        current = os.environ.get("MIC_DEVICE", "")
        default_item = rumps.MenuItem("System Default", callback=self._select_mic)
        default_item.state = 1 if not current else 0
        items = [default_item, None]
        for _, name in list_input_devices():
            item = rumps.MenuItem(name, callback=self._select_mic)
            item.state = 1 if (current and current.lower() in name.lower()) else 0
            items.append(item)
        self.mic_menu.update(items)

    def _select_mic(self, sender):
        for item in self.mic_menu.values():
            if hasattr(item, "state"):
                item.state = 0
        sender.state = 1
        value = "" if sender.title == "System Default" else sender.title
        os.environ["MIC_DEVICE"] = value
        save_env("MIC_DEVICE", value)

        # Hot-swap mic on the wake detector if currently listening
        if self.is_listening and self.wake_detector:
            mic_device = None
            if value:
                mic_device, found_name = find_input_device(value)
                if found_name:
                    print(f"✓ Switching mic to: [{mic_device}] {found_name}")
            else:
                print("✓ Switching mic to: System Default")
            self.wake_detector.device = mic_device
            self.wake_detector.stop_stream()
            self.wake_detector.start_stream()

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

        mic_name = os.environ.get("MIC_DEVICE", "")
        mic_device = None
        if mic_name:
            mic_device, found_name = find_input_device(mic_name)
            if found_name:
                print(f"✓ Using mic device: [{mic_device}] {found_name}")
            else:
                print(f"⚠ Mic device '{mic_name}' not found — using system default")

        self._last_default_input = sd.default.device[0]

        if porcupine_key and porcupine_model:
            print("✓ Using Porcupine wake word detector (on-device, fast)")
            self.wake_detector = PorcupineWakeWordDetector(
                access_key=porcupine_key,
                keyword_path=porcupine_model,
                callback=self.on_wake_word,
                device=mic_device,
            )
        else:
            print("⚠ Porcupine not configured — using energy-based fallback detector.")
            print("  For better accuracy, see README for Porcupine setup.")
            self.wake_detector = FallbackWakeWordDetector(
                callback=self.on_wake_word,
                device=mic_device,
            )

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
        control_media("pause")
        audio_snapshot = self.audio_buffer.get_audio()
        buffer_seconds = len(audio_snapshot) / SAMPLE_RATE
        print(f"Audio snapshot: {buffer_seconds:.1f}s")

        if buffer_seconds < 5:
            speak("I don't have enough audio context yet. Try again after listening for a bit.")
            control_media("play")
            self.is_explaining = False
            self.set_status("Buffering locally...", "🔴")
            return

        # Start Whisper on the buffer immediately — runs during command recording
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        print(f"Submitting {buffer_seconds:.0f}s buffer to Whisper...")
        future_transcript = pool.submit(self.transcriber.transcribe, audio_snapshot)

        # Prime the capture queue before the chime so no audio is missed
        if self.wake_detector:
            self.wake_detector.start_capture()

        self.set_status("Listening...", "👂")
        subprocess.Popen(["afplay", "/System/Library/Sounds/Tink.aiff"])

        user_command_audio = self.wake_detector.get_capture_audio(
            max_duration=CAPTURE_USER_COMMAND_DURATION,
            pre_speech_timeout=2.0
        )

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
            speak("Sorry, transcription failed. Check your API key and internet connection.")
            control_media("play")
            self.is_explaining = False
            self.set_status("Buffering locally...", "🔴")
            return

        if not transcript.strip():
            speak("Couldn't make out what was being said. Check your Screen Recording permission.")
            control_media("play")
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
            speak_stream(self.explainer.explain_audio_stream(transcript, focus=focus))
        except Exception as e:
            print(f"Explanation error: {e}")
            speak("Sorry, I had trouble generating an explanation.")
            control_media("play")
            self.is_explaining = False
            self.set_status("Buffering locally...", "🔴")
            return

        # Resume playback
        time.sleep(0.3)
        control_media("play")

        self.is_explaining = False
        self.set_status("Buffering locally...", "🔴")

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
            save_env("OPENAI_API_KEY", response.text)
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
            save_env("PORCUPINE_ACCESS_KEY", response2.text)

        response3 = rumps.Window(
            message="Custom Porcupine model path (.ppn file — leave blank to use default):",
            title="Podcast Copilot — Settings",
            default_text=os.environ.get("PORCUPINE_MODEL_PATH", ""),
            ok="Save",
            cancel="Skip",
            dimensions=(420, 30)
        ).run()
        if response3.clicked and response3.text:
            os.environ["PORCUPINE_MODEL_PATH"] = response3.text
            save_env("PORCUPINE_MODEL_PATH", response3.text)


def main():
    load_env()
    app = PodcastCopilot()
    app.run()


if __name__ == "__main__":
    main()
