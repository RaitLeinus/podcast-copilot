"""
fallback.py - Energy-based speech detector, no setup required.

No wake word recognition — triggers when sustained speech is detected on the mic.
Good enough for solo listening sessions in quiet environments.
For shared spaces or better accuracy, use PorcupineWakeWordDetector instead.
"""

import threading
import time
import queue
import collections
import numpy as np
import sounddevice as sd


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
    CHUNK_SECONDS = 0.05        # 50ms chunks — shared by detection and capture

    # How many chunks to keep as pre-trigger history (2s at 50ms/chunk = 40)
    PRE_BUFFER_CHUNKS = 40

    def __init__(self, callback, device=None):
        self.callback = callback
        self.device = device
        self._running = False
        self._stream = None
        self._last_triggered = 0.0
        self._speech_duration = 0.0
        self._capturing = False
        self._capture_queue = queue.Queue()
        self._pre_buffer = collections.deque(maxlen=self.PRE_BUFFER_CHUNKS)
        print("⚠  Using fallback energy-based wake detector.")
        print("   Speak for ~1.5 seconds to trigger an explanation.")
        print("   For real wake word detection, set up Porcupine (see README).")

    def start(self):
        self._running = True
        self.start_stream()

    def start_stream(self):
        """Start (or restart) the mic audio stream."""
        chunk_size = int(self.SAMPLE_RATE * self.CHUNK_SECONDS)
        self._stream = sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=chunk_size,
            device=self.device,
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop_stream(self):
        """Stop just the mic audio stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def stop(self):
        self._running = False
        self.stop_stream()

    def start_capture(self):
        """Begin buffering mic audio for command capture (stream already running).
        Pre-fills the capture queue from the first speech onset in the pre-buffer,
        so the VAD doesn't hit pre_speech_timeout on leading silence.
        """
        while not self._capture_queue.empty():
            try:
                self._capture_queue.get_nowait()
            except queue.Empty:
                break
        pre_buffer_list = list(self._pre_buffer)
        speech_start = None
        for i, chunk in enumerate(pre_buffer_list):
            if float(np.sqrt(np.mean(chunk ** 2))) > self.ENERGY_THRESHOLD:
                speech_start = i
                break
        if speech_start is not None:
            for chunk in pre_buffer_list[speech_start:]:
                self._capture_queue.put(chunk)
        self._capturing = True

    def get_capture_audio(self, max_duration=4.0, silence_threshold=0.015,
                          silence_seconds=0.6, pre_speech_timeout=0.8):
        """
        Collect audio from the shared mic stream with VAD.
        Returns float32 ndarray. Call start_capture() before playing the chime.
        """
        chunk_seconds = self.CHUNK_SECONDS
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
                chunk = self._capture_queue.get(timeout=0.5)
            except queue.Empty:
                break
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
        chunk = indata[:, 0].copy()

        # Always maintain a rolling pre-trigger history
        self._pre_buffer.append(chunk)

        # Wake word detection (energy-based)
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        if rms > self.ENERGY_THRESHOLD:
            self._speech_duration += self.CHUNK_SECONDS
        else:
            self._speech_duration = 0.0

        if self._speech_duration >= self.TRIGGER_SECONDS:
            self._speech_duration = 0.0
            now = time.time()
            if now - self._last_triggered >= self.COOLDOWN_SECONDS:
                self._last_triggered = now
                print(f"Fallback detector: speech detected ({rms:.4f} RMS) — triggering")
                threading.Thread(target=self.callback, daemon=True).start()

        # Capture routing
        if self._capturing:
            self._capture_queue.put(chunk)
