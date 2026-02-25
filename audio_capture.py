"""
audio_capture.py - Captures system audio via BlackHole virtual audio device.

BlackHole routes whatever is playing on your Mac to a virtual input device
that this app reads from. Audio is passed directly to RollingAudioBuffer
and never sent anywhere externally.

Setup: Install BlackHole (brew install blackhole-2ch), then create a
Multi-Output Device in Audio MIDI Setup that includes both BlackHole and
your normal speakers/headphones. Set that as your Mac's sound output.
"""

import sounddevice as sd
import numpy as np


def find_blackhole_device():
    """Find BlackHole virtual audio input device index."""
    for i, device in enumerate(sd.query_devices()):
        if "blackhole" in device.get("name", "").lower():
            if device.get("max_input_channels", 0) > 0:
                return i, device["name"]
    return None, None


def list_input_devices():
    return [
        (i, d["name"])
        for i, d in enumerate(sd.query_devices())
        if d.get("max_input_channels", 0) > 0
    ]


class AudioCapture:
    def __init__(self, sample_rate: int = 16000, chunk_seconds: float = 0.5):
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_seconds)
        self._stream = None

    def start(self, callback):
        """
        Start capturing audio.
        callback(chunk: np.ndarray) is called with each audio chunk.
        Chunks are float32 mono arrays at self.sample_rate.
        """
        device_index, device_name = find_blackhole_device()

        if device_index is None:
            print("⚠  BlackHole not found — falling back to default input.")
            print("   System audio capture won't work until BlackHole is set up.")
            print("   Install: brew install blackhole-2ch")
            print("   Then set up a Multi-Output Device in Audio MIDI Setup.\n")
            print("   Available input devices:")
            for idx, name in list_input_devices():
                print(f"     [{idx}] {name}")
        else:
            print(f"✓ Capturing system audio via: [{device_index}] {device_name}")

        self._stream = sd.InputStream(
            device=device_index,
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_size,
            callback=lambda indata, frames, t, status: callback(indata[:, 0].copy())
        )
        self._stream.start()

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
