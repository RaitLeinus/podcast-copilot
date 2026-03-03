/// audio_capture_lib — Native ScreenCaptureKit audio capture, loadable via ctypes.
///
/// Build: swiftc -O -emit-library -framework ScreenCaptureKit -framework CoreMedia -module-name AudioCaptureHelper -o libaudio_capture.dylib capture_helper.swift
///
/// C API for Python:
///   sck_start_capture(sample_rate, error_buf, error_buf_size) -> Int32  (0=ok, -1=error)
///   sck_stop_capture()
///   sck_read_audio(out_buf, max_bytes) -> Int32  (bytes copied)

import Foundation
import ScreenCaptureKit
import CoreMedia

// MARK: - Ring buffer for captured audio

private let bufferLock = NSLock()
private var ringBuffer = Data()
private let maxBufferSize = 16000 * 4 * 120  // 120 seconds of float32 at 16kHz

// MARK: - SCStreamOutput delegate

private class AudioOutputDelegate: NSObject, SCStreamOutput {
    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .audio else { return }
        guard let floats = extractFloat32(from: sampleBuffer), !floats.isEmpty else { return }

        let bytes = floats.withUnsafeBufferPointer { Data(buffer: $0) }
        bufferLock.lock()
        ringBuffer.append(bytes)
        if ringBuffer.count > maxBufferSize {
            ringBuffer.removeFirst(ringBuffer.count - maxBufferSize)
        }
        bufferLock.unlock()
    }
}

private func extractFloat32(from sampleBuffer: CMSampleBuffer) -> [Float]? {
    guard let blockBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) else { return nil }
    var length = 0
    var dataPointer: UnsafeMutablePointer<Int8>?
    let status = CMBlockBufferGetDataPointer(blockBuffer, atOffset: 0, lengthAtOffsetOut: nil, totalLengthOut: &length, dataPointerOut: &dataPointer)
    guard status == noErr, let data = dataPointer, length > 0 else { return nil }

    let floatCount = length / MemoryLayout<Float>.size
    let floatPtr = UnsafeRawPointer(data).bindMemory(to: Float.self, capacity: floatCount)
    return Array(UnsafeBufferPointer(start: floatPtr, count: floatCount))
}

// MARK: - Shared state

private var activeStream: SCStream?
private var activeDelegate: AudioOutputDelegate?
private var captureStarted = false

// MARK: - C API (called from Python via ctypes)

@_cdecl("sck_start_capture")
public func sck_start_capture(_ sampleRate: Int32, _ errorBuf: UnsafeMutablePointer<CChar>?, _ errorBufSize: Int32) -> Int32 {
    guard !captureStarted else { return 0 }

    let semaphore = DispatchSemaphore(value: 0)
    var resultCode: Int32 = 0

    Task {
        do {
            let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
            guard let display = content.displays.first else {
                writeError("No displays found. Grant Screen Recording permission.", to: errorBuf, size: errorBufSize)
                resultCode = -1
                semaphore.signal()
                return
            }

            let config = SCStreamConfiguration()
            config.capturesAudio = true
            config.sampleRate = Int(sampleRate)
            config.channelCount = 1
            config.excludesCurrentProcessAudio = true

            let filter = SCContentFilter(display: display, excludingWindows: [])
            let stream = SCStream(filter: filter, configuration: config, delegate: nil)
            let delegate = AudioOutputDelegate()
            try stream.addStreamOutput(delegate, type: .audio, sampleHandlerQueue: .global(qos: .userInteractive))
            try await stream.startCapture()

            activeStream = stream
            activeDelegate = delegate
            captureStarted = true
            semaphore.signal()
        } catch {
            writeError(error.localizedDescription, to: errorBuf, size: errorBufSize)
            resultCode = -1
            semaphore.signal()
        }
    }

    semaphore.wait()
    return resultCode
}

@_cdecl("sck_stop_capture")
public func sck_stop_capture() {
    guard captureStarted, let stream = activeStream else { return }
    let semaphore = DispatchSemaphore(value: 0)
    Task {
        try? await stream.stopCapture()
        semaphore.signal()
    }
    _ = semaphore.wait(timeout: .now() + 5.0)
    activeStream = nil
    activeDelegate = nil
    captureStarted = false
    bufferLock.lock()
    ringBuffer.removeAll()
    bufferLock.unlock()
}

/// Read available audio data from the ring buffer.
/// Copies up to maxBytes into outBuf. Returns bytes actually copied.
@_cdecl("sck_read_audio")
public func sck_read_audio(_ outBuf: UnsafeMutablePointer<UInt8>, _ maxBytes: Int32) -> Int32 {
    bufferLock.lock()
    let available = min(Int(maxBytes), ringBuffer.count)
    if available > 0 {
        ringBuffer.copyBytes(to: outBuf, count: available)
        ringBuffer.removeFirst(available)
    }
    bufferLock.unlock()
    return Int32(available)
}

private func writeError(_ msg: String, to buf: UnsafeMutablePointer<CChar>?, size: Int32) {
    guard let buf, size > 0 else { return }
    let bytes = Array(msg.utf8.prefix(Int(size) - 1))
    for (i, b) in bytes.enumerated() {
        buf[i] = CChar(bitPattern: b)
    }
    buf[bytes.count] = 0
}
