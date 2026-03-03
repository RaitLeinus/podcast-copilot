"""
audio_buffer.py - Rolling in-memory audio buffer.

Stores the last N seconds of audio as raw float32 samples.
Nothing leaves this buffer until you explicitly call get_audio().
Thread-safe: audio capture and wake word threads both touch this.
"""

import threading
import numpy as np
from collections import deque


class RollingAudioBuffer:
    def __init__(self, sample_rate: int, max_seconds: int = 120):
        """
        sample_rate: samples per second (e.g. 16000)
        max_seconds: how many seconds of audio to keep (older audio is dropped)
        """
        self.sample_rate = sample_rate
        self.max_samples = sample_rate * max_seconds
        self._buffer = deque(maxlen=self.max_samples)
        self._lock = threading.Lock()
        self._debug_logged = False

    def append(self, chunk: np.ndarray):
        """Add a chunk of audio samples to the buffer. Thread-safe."""
        if not self._debug_logged and len(chunk) > 0:
            print(f"[DEBUG] First audio chunk received: {len(chunk)} samples, dtype={chunk.dtype}")
            self._debug_logged = True
        with self._lock:
            self._buffer.extend(chunk)

    def get_audio(self) -> np.ndarray:
        """
        Return a copy of all buffered audio as a float32 numpy array.
        Safe to call from any thread — takes a snapshot, doesn't clear.
        """
        with self._lock:
            return np.array(self._buffer, dtype=np.float32)

    def clear(self):
        """Discard all buffered audio."""
        with self._lock:
            self._buffer.clear()

    @property
    def duration_seconds(self) -> float:
        """How many seconds of audio are currently buffered."""
        with self._lock:
            return len(self._buffer) / self.sample_rate

    @property
    def sample_count(self) -> int:
        with self._lock:
            return len(self._buffer)
