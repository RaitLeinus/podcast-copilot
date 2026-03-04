"""
capture.py - Captures system audio via a native Swift dylib.

The dylib uses ScreenCaptureKit directly (no pyobjc needed) and runs
in-process so it inherits the terminal's Screen Recording permission.

Build the dylib (one time, from the audio/ directory):
  swiftc -O -emit-library -target x86_64-apple-macosx14.0 \
    -framework ScreenCaptureKit -framework CoreMedia \
    -module-name AudioCaptureHelper \
    -o libaudio_capture.dylib capture_helper.swift

Requires one-time Screen Recording permission grant:
  System Settings → Privacy & Security → Screen Recording → enable your terminal app.
"""

import ctypes
import os
import threading
import time

import numpy as np
import sounddevice as sd


def find_input_device(name: str):
    """Return (index, full_name) of first input device matching name (case-insensitive substring)."""
    for i, device in enumerate(sd.query_devices()):
        if device.get("max_input_channels", 0) > 0 and name.lower() in device.get("name", "").lower():
            return i, device["name"]
    return None, None



def list_input_devices():
    return [
        (i, d["name"])
        for i, d in enumerate(sd.query_devices())
        if d.get("max_input_channels", 0) > 0
    ]


def _load_dylib():
    """Load the native ScreenCaptureKit dylib."""
    dylib_path = os.path.join(os.path.dirname(__file__), "libaudio_capture.dylib")
    if not os.path.isfile(dylib_path):
        raise RuntimeError(
            f"libaudio_capture.dylib not found at {dylib_path}.\n"
            "Build it from the audio/ directory:\n"
            "  swiftc -O -emit-library -target x86_64-apple-macosx14.0 "
            "-framework ScreenCaptureKit -framework CoreMedia "
            "-module-name AudioCaptureHelper "
            "-o libaudio_capture.dylib capture_helper.swift"
        )
    lib = ctypes.CDLL(dylib_path)
    lib.sck_start_capture.restype = ctypes.c_int32
    lib.sck_start_capture.argtypes = [ctypes.c_int32, ctypes.c_char_p, ctypes.c_int32]
    lib.sck_stop_capture.restype = None
    lib.sck_read_audio.restype = ctypes.c_int32
    lib.sck_read_audio.argtypes = [ctypes.c_char_p, ctypes.c_int32]
    return lib


class AudioCapture:
    """Captures system audio via native ScreenCaptureKit dylib loaded in-process."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._lib = None
        self._running = False
        self._reader_thread = None

    def start(self, callback):
        """
        Start capturing system audio.
        callback(chunk: np.ndarray) is called with float32 mono chunks at self.sample_rate.
        """
        self._lib = _load_dylib()

        err_buf = ctypes.create_string_buffer(512)
        result = self._lib.sck_start_capture(self.sample_rate, err_buf, 512)
        if result != 0:
            error_msg = err_buf.value.decode("utf-8", errors="replace")
            raise RuntimeError(f"ScreenCaptureKit: {error_msg}")

        print(f"✓ ScreenCaptureKit: capturing system audio at {self.sample_rate} Hz")

        self._running = True

        # Poll the dylib's ring buffer and deliver chunks to callback
        def _reader():
            read_buf = ctypes.create_string_buffer(self.sample_rate * 4)  # 1 second
            while self._running:
                n = self._lib.sck_read_audio(read_buf, len(read_buf))
                if n > 0:
                    chunk = np.frombuffer(read_buf.raw[:n], dtype=np.float32).copy()
                    callback(chunk)
                else:
                    time.sleep(0.05)  # no data yet, wait briefly

        self._reader_thread = threading.Thread(target=_reader, daemon=True)
        self._reader_thread.start()

    def stop(self):
        self._running = False
        if self._lib:
            self._lib.sck_stop_capture()
            self._lib = None
