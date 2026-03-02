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
        self._stream = None
        self._capturing = False
        self._capture_queue = queue.Queue()
        print(f"✓ Porcupine loaded — wake word model: {keyword_path}")

    def start(self):
        self._running = True
        frame_length = self.porcupine.frame_length
        sample_rate = self.porcupine.sample_rate
        print(f"✓ Porcupine listening on mic (frame={frame_length}, sr={sample_rate})")
        self._stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=frame_length,
            device=None,
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self):
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if hasattr(self, "porcupine"):
            self.porcupine.delete()

    def start_capture(self):
        """Begin buffering mic audio for command capture (stream stays open)."""
        while not self._capture_queue.empty():
            try:
                self._capture_queue.get_nowait()
            except queue.Empty:
                break
        self._capturing = True

    def get_capture_audio(self, max_duration=4.0, silence_threshold=0.015,
                          silence_seconds=0.6, pre_speech_timeout=0.8):
        """
        Collect audio from the shared mic stream with VAD.
        Returns float32 ndarray. Call start_capture() before playing the chime.
        """
        frame_length = self.porcupine.frame_length
        sample_rate = self.porcupine.sample_rate
        chunk_seconds = frame_length / sample_rate

        silence_needed = int(silence_seconds / chunk_seconds)
        pre_speech_timeout_chunks = int(pre_speech_timeout / chunk_seconds)
        max_chunks = int(max_duration / chunk_seconds)
        speech_confirm_needed = int(0.15 / chunk_seconds)

        recorded = []
        speech_started = False
        speech_confirm_count = 0
        silence_count = 0
        pre_speech_count = 0

        for _ in range(max_chunks):
            try:
                pcm = self._capture_queue.get(timeout=0.5)
            except queue.Empty:
                break
            chunk = pcm.astype(np.float32) / 32768.0
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
                if pre_speech_count >= pre_speech_timeout_chunks:
                    print("VAD: no speech detected, stopping early")
                    break

        self._capturing = False
        return np.concatenate(recorded) if recorded else np.zeros(0, dtype=np.float32)

    def _audio_callback(self, indata, _frames, _time_info, _status):
        pcm = indata[:, 0].copy()
        result = self.porcupine.process(pcm)
        if result >= 0:
            print("✓ Porcupine: wake word detected!")
            self.callback()
        if self._capturing:
            self._capture_queue.put(pcm)


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
