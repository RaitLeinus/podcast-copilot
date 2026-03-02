"""
wake_word_porcupine.py - On-device wake word detection via Picovoice Porcupine.
Detects a custom wake word (e.g. "explain that") with ~50ms latency.
Runs entirely on-device — no audio ever leaves your machine.

Setup:
1. pip install pvporcupine
2. Free account at https://console.picovoice.ai/ → get Access Key
3. Create a custom wake word in the Picovoice Console → download .ppn file
4. Set env vars:
     PORCUPINE_ACCESS_KEY=your-key
     PORCUPINE_MODEL_PATH=/path/to/your_wake_word.ppn
"""

import queue
import numpy as np
import sounddevice as sd


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
