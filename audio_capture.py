"""
audio_capture.py - Captures system audio via ScreenCaptureKit.

ScreenCaptureKit (macOS 13+) captures system audio directly —
no virtual audio device needed. Requires one-time Screen Recording
permission grant and pyobjc-framework-ScreenCaptureKit.
"""

import platform
import threading
import ctypes
import ctypes.util

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


class _SCKAudioCapture:
    """
    System audio capture via ScreenCaptureKit.

    Requires one-time Screen Recording permission:
    System Settings → Privacy & Security → Screen Recording → enable your terminal/app.

    Dependency: pip install pyobjc-framework-ScreenCaptureKit
    """

    def __init__(self, sample_rate: int, callback):
        self._sample_rate = sample_rate
        self._callback = callback
        self._sck_stream = None
        self._delegate = None

    def start(self):
        import warnings
        import objc
        from Foundation import NSObject
        import ScreenCaptureKit as SCKit

        # CMSampleBufferRef is an opaque CF type; pyobjc wraps it as PyObjCPointer
        # (^v void*) which is correct but triggers a harmless informational warning.
        warnings.filterwarnings("ignore", category=objc.ObjCPointerWarning)

        sample_rate = self._sample_rate
        callback = self._callback

        # ── CMSampleBuffer → float32 numpy (via ctypes / CoreMedia) ────────────
        _cm = ctypes.CDLL(ctypes.util.find_library("CoreMedia"))
        _cf = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))
        # Must declare argtypes: parameter 4 is size_t (64-bit); without this ctypes
        # passes it as 32-bit int, corrupting the stack and causing a segfault.
        _cm.CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer.restype = ctypes.c_int32
        _cm.CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer.argtypes = [
            ctypes.c_void_p,                   # CMSampleBufferRef
            ctypes.c_void_p,                   # size_t *sizeNeededOut (NULL ok)
            ctypes.c_void_p,                   # AudioBufferList *
            ctypes.c_size_t,                   # size_t audioBufferListSize  ← 64-bit!
            ctypes.c_void_p,                   # CFAllocatorRef (NULL = default)
            ctypes.c_void_p,                   # CFAllocatorRef (NULL = default)
            ctypes.c_uint32,                   # uint32_t flags
            ctypes.POINTER(ctypes.c_void_p),   # CMBlockBufferRef *
        ]
        _cf.CFRelease.argtypes = [ctypes.c_void_p]

        class _AudioBuffer(ctypes.Structure):
            _fields_ = [
                ("mNumberChannels", ctypes.c_uint32),
                ("mDataByteSize", ctypes.c_uint32),
                ("mData", ctypes.c_void_p),
            ]

        class _AudioBufferList(ctypes.Structure):
            _fields_ = [
                ("mNumberBuffers", ctypes.c_uint32),
                ("mBuffers", _AudioBuffer * 8),
            ]

        def _extract_pcm(sample_buffer_ptr):
            # First query the required AudioBufferList size.
            size_needed = ctypes.c_size_t(0)
            _cm.CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
                sample_buffer_ptr, ctypes.byref(size_needed),
                None, 0,
                None, None, 0,
                None,
            )
            if size_needed.value == 0:
                return None

            # Allocate a raw buffer of the right size and cast to AudioBufferList.
            abl_buf = (ctypes.c_byte * size_needed.value)()
            abl_ptr = ctypes.cast(abl_buf, ctypes.POINTER(_AudioBufferList))
            block_buf = ctypes.c_void_p()
            status = _cm.CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
                sample_buffer_ptr, None,
                abl_ptr, size_needed.value,
                None, None, 0,
                ctypes.byref(block_buf),
            )
            if status != 0:
                return None
            abl = abl_ptr.contents
            chunks = []
            for i in range(min(abl.mNumberBuffers, 8)):
                buf = abl.mBuffers[i]
                if buf.mData and buf.mDataByteSize > 0:
                    raw = ctypes.string_at(buf.mData, buf.mDataByteSize)
                    chunks.append(np.frombuffer(raw, dtype=np.float32).copy())
            if block_buf.value:
                _cf.CFRelease(block_buf)
            if not chunks:
                return None
            return chunks[0] if len(chunks) == 1 else np.column_stack(chunks).mean(axis=1)

        # ── SCStreamOutput delegate ─────────────────────────────────────────────
        # CMSampleBufferRef is a CFTypeRef — the ObjC runtime passes it as an
        # object reference (@), not a raw void pointer (^v).  Using ^v corrupts
        # argument marshalling and causes a segfault.
        class _AudioDelegate(NSObject):
            @objc.typedSelector(b"v@:@@q")
            def stream_didOutputSampleBuffer_ofType_(self, _stream, sample_buffer, output_type):
                # output_type: SCStreamOutputTypeScreen=0, SCStreamOutputTypeAudio=1
                if output_type != 1:
                    return
                try:
                    sb_ptr = ctypes.c_void_p(sample_buffer.pointerAsInteger)
                    chunk = _extract_pcm(sb_ptr)
                    if chunk is not None and len(chunk) > 0:
                        callback(chunk)
                except Exception:
                    import traceback
                    traceback.print_exc()

        delegate = _AudioDelegate.alloc().init()
        self._delegate = delegate

        # ── Get shareable content ───────────────────────────────────────────────
        result = {}
        done = threading.Event()

        def _content_handler(content, error):
            result["content"] = content
            result["error"] = error
            done.set()

        SCKit.SCShareableContent.getShareableContentWithCompletionHandler_(_content_handler)
        if not done.wait(timeout=10):
            raise RuntimeError("ScreenCaptureKit: timed out waiting for shareable content")

        if result.get("error"):
            raise RuntimeError(
                f"ScreenCaptureKit unavailable: {result['error']}\n"
                "Grant Screen Recording access in System Settings → Privacy & Security → Screen Recording."
            )

        content = result["content"]
        displays = content.displays()
        if not displays:
            raise RuntimeError("ScreenCaptureKit: no displays found")

        # ── Configure audio stream ──────────────────────────────────────────────
        # A display filter is required even for audio-only capture.
        config = SCKit.SCStreamConfiguration.alloc().init()
        config.setCapturesAudio_(True)
        config.setSampleRate_(float(sample_rate))
        config.setChannelCount_(1)
        try:
            config.setExcludesCurrentProcessAudio_(True)  # macOS 14+ only
        except AttributeError:
            pass

        content_filter = SCKit.SCContentFilter.alloc().initWithDisplay_excludingWindows_(
            displays[0], []
        )
        stream = SCKit.SCStream.alloc().initWithFilter_configuration_delegate_(
            content_filter, config, objc.nil
        )

        # SCStreamOutputTypeAudio = 1; None queue → SCKit's private serial queue
        ok = stream.addStreamOutput_type_sampleHandlerQueue_error_(
            delegate, 1, None, None
        )
        if not ok:
            raise RuntimeError("ScreenCaptureKit: failed to add audio stream output")

        # ── Start capture ───────────────────────────────────────────────────────
        start_done = threading.Event()
        start_err = {}

        def _start_handler(error):
            if error:
                start_err["e"] = error
            start_done.set()

        stream.startCaptureWithCompletionHandler_(_start_handler)
        if not start_done.wait(timeout=10):
            raise RuntimeError("ScreenCaptureKit: timed out starting capture")
        if "e" in start_err:
            raise RuntimeError(f"ScreenCaptureKit: stream failed to start: {start_err['e']}")

        self._sck_stream = stream
        print(f"✓ ScreenCaptureKit: capturing system audio at {sample_rate} Hz")

    def stop(self):
        if self._sck_stream is None:
            return
        done = threading.Event()
        self._sck_stream.stopCaptureWithCompletionHandler_(lambda err: done.set())
        done.wait(timeout=5.0)
        self._sck_stream = None
        self._delegate = None


class AudioCapture:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._backend = None

    def start(self, callback):
        """
        Start capturing system audio via ScreenCaptureKit.
        callback(chunk: np.ndarray) is called with float32 mono chunks at self.sample_rate.
        """
        parts = platform.mac_ver()[0].split(".")
        if not parts or int(parts[0]) < 13:
            raise RuntimeError("ScreenCaptureKit requires macOS 13+")
        try:
            import ScreenCaptureKit  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "pyobjc-framework-ScreenCaptureKit not installed.\n"
                "Run: pip install pyobjc-framework-ScreenCaptureKit"
            )

        backend = _SCKAudioCapture(self.sample_rate, callback)
        backend.start()
        self._backend = backend

    def stop(self):
        if self._backend:
            self._backend.stop()
            self._backend = None
