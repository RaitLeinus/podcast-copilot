"""
wake_word.py - On-device wake word detection. No API calls, runs entirely locally.

Two implementations:
  PorcupineWakeWordDetector  — recommended, uses Picovoice Porcupine (~50ms latency)
  FallbackWakeWordDetector   — no setup needed, detects sustained speech via mic energy

The fallback is a simple "I heard you speak" detector — it doesn't understand words,
just detects when you speak for ~1.5 seconds. Good enough for solo listening sessions
where background noise is low. For shared spaces, use Porcupine.
"""

import threading
import time
import queue
import numpy as np
import sounddevice as sd


# ─── Porcupine (recommended) ────────────────────────────────────────────────

class PorcupineWakeWordDetector:
    """
    On-device wake word detection via Picovoice Porcupine.
    Detects a custom wake word (e.g. "hey copilot") with ~50ms latency.
    Runs entirely on-device — no audio ever leaves your machine.

    Setup:
    1. pip install pvporcupine
    2. Free account at https://console.picovoice.ai/ → get Access Key
    3. Create a custom wake word in the Picovoice Console → download .ppn file
    4. Set env vars:
         PORCUPINE_ACCESS_KEY=your-key
         PORCUPINE_MODEL_PATH=/path/to/your_wake_word.ppn
    """

    def __init__(self, access_key: str, keyword_path: str, callback):
        """
        access_key:   Picovoice access key
        keyword_path: path to .ppn wake word model file
        callback:     function() called when wake word is detected
        """
        try:
            import pvporcupine
        except ImportError:
            raise ImportError(
                "pvporcupine not installed.\n"
                "Run: pip install pvporcupine\n"
                "Then get a free key at https://console.picovoice.ai/"
            )

        import pvporcupine
        self.porcupine = pvporcupine.create(
            access_key=access_key,
            keyword_paths=[keyword_path]
        )
        self.callback = callback
        self._running = False
        self._paused = False
        self._thread = None
        print(f"✓ Porcupine loaded — wake word model: {keyword_path}")

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if hasattr(self, "porcupine"):
            self.porcupine.delete()

    def pause(self):
        """Close the mic stream temporarily (e.g. while command is being recorded)."""
        self._paused = True

    def resume(self):
        self._paused = False

    def _run(self):
        frame_length = self.porcupine.frame_length
        sample_rate = self.porcupine.sample_rate

        while self._running:
            if self._paused:
                time.sleep(0.05)
                continue

            print(f"✓ Porcupine listening on mic (frame={frame_length}, sr={sample_rate})")
            with sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="int16",
                blocksize=frame_length,
                device=None
            ) as stream:
                while self._running and not self._paused:
                    pcm, _ = stream.read(frame_length)
                    pcm_flat = pcm.flatten()
                    result = self.porcupine.process(pcm_flat)
                    if result >= 0:
                        print("✓ Porcupine: wake word detected!")
                        self.callback()


# ─── Fallback (no setup needed) ─────────────────────────────────────────────

class FallbackWakeWordDetector:
    """
    Simple energy-based speech detector.
    No wake word recognition — just detects when you start speaking.
    Triggers after TRIGGER_SECONDS of sustained speech above an energy threshold.

    How to use: just speak normally near your microphone.
    A short "hey" or any ~1.5 second utterance will trigger it.

    Limitations:
    - Will trigger on any voice (yours, podcast guests, room noise)
    - For best results: mute your mic to the podcast app, use in quiet environment
    - Upgrade to Porcupine for a real wake word
    """

    ENERGY_THRESHOLD = 0.01     # RMS energy to consider "speech" (tune if needed)
    TRIGGER_SECONDS = 0.5       # How long speech must be sustained to trigger
    COOLDOWN_SECONDS = 3.0      # Minimum time between triggers
    SAMPLE_RATE = 16000
    CHUNK_SECONDS = 0.1

    def __init__(self, callback):
        self.callback = callback
        self._running = False
        self._paused = False
        self._thread = None
        self._last_triggered = 0.0
        print("⚠  Using fallback energy-based wake detector.")
        print("   Speak for ~1.5 seconds to trigger an explanation.")
        print("   For real wake word detection, set up Porcupine (see README).")

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def pause(self):
        """Close the mic stream temporarily (e.g. while command is being recorded)."""
        self._paused = True

    def resume(self):
        self._paused = False

    def _run(self):
        chunk_size = int(self.SAMPLE_RATE * self.CHUNK_SECONDS)

        while self._running:
            if self._paused:
                time.sleep(0.05)
                continue

            speech_duration = 0.0
            audio_q = queue.Queue()

            def mic_callback(indata, frames, t, status):
                audio_q.put(indata[:, 0].copy())

            with sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=chunk_size,
                device=None,
                callback=mic_callback
            ):
                while self._running and not self._paused:
                    try:
                        chunk = audio_q.get(timeout=0.5)
                        rms = float(np.sqrt(np.mean(chunk ** 2)))

                        if rms > self.ENERGY_THRESHOLD:
                            speech_duration += self.CHUNK_SECONDS
                        else:
                            speech_duration = 0.0  # reset on silence

                        if speech_duration >= self.TRIGGER_SECONDS:
                            speech_duration = 0.0
                            now = time.time()
                            if now - self._last_triggered >= self.COOLDOWN_SECONDS:
                                self._last_triggered = now
                                print(f"Fallback detector: speech detected ({rms:.4f} RMS) — triggering")
                                self.callback()

                    except queue.Empty:
                        speech_duration = 0.0
                        continue
