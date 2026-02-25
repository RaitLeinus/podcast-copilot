# 🎙 Podcast Copilot

A Mac menu bar app that listens to whatever you're playing (Spotify, YouTube, any app) and explains content on voice command. Say **"explain that"** and it pauses, explains what you just heard, then resumes.

## How It Works

```
System audio (Spotify/YouTube/Chrome)
        ↓
  BlackHole virtual audio device
        ↓
  Rolling 2-min audio buffer  ← stored in RAM only, nothing sent anywhere
        ↓ (only when wake word fires)
  Whisper API  ←  first API call, only on demand
        ↓
  GPT-4o  ←  second API call
        ↓
  macOS "say" speaks the explanation aloud
        ↓
  Resume playback


Microphone → Porcupine (on-device) → wake word trigger
             (no audio leaves your machine)
```

**Privacy model:** The podcast audio never leaves your computer until you explicitly ask for an explanation. The wake word runs 100% on-device. Only when you say "explain that" does ~2 minutes of buffered audio get sent to Whisper.

---

## Setup

### 1. Install BlackHole (system audio capture)

```bash
brew install blackhole-2ch
```

Or download from https://existential.audio/blackhole/

**Create a Multi-Output Device so you still hear audio:**
1. Open **Audio MIDI Setup** (Applications → Utilities)
2. Click **+** → **Create Multi-Output Device**
3. Check both **BlackHole 2ch** and your normal output (speakers/headphones)
4. Right-click the Multi-Output Device → **Use This Device for Sound Output**

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up Porcupine wake word (recommended)

Porcupine runs fully on-device with ~50ms latency.

1. Free account at https://console.picovoice.ai/ → copy your **Access Key**
2. Go to **Console → Wake Word** → create a new wake word (e.g. "hey copilot" or "explain that")
3. Download the generated `.ppn` model file for Mac
4. Set environment variables:
   ```bash
   export PORCUPINE_ACCESS_KEY=your-access-key
   export PORCUPINE_MODEL_PATH=/path/to/hey_copilot_mac.ppn
   ```

> **No Porcupine?** The app falls back to energy-based detection — it triggers when you speak for ~1.5 seconds. Works fine in quiet environments but has no actual phrase recognition.

### 4. Set your OpenAI API key

```bash
export OPENAI_API_KEY=sk-...
```

Or enter it via the ⚙ Settings menu after launch.

---

## Running

```bash
python app.py
```

A 🎙 icon appears in the menu bar. Click **▶ Start Listening**, then play any podcast or video.

The menu shows a live buffer counter (e.g. "Buffer: 1m 45s captured") so you know how much context is available.

**To get an explanation:** say your wake word (or speak for ~1.5s if using fallback). The app will:
1. Immediately pause playback
2. Send the buffered audio to Whisper (~2–5 seconds)
3. Send the transcript to GPT-4o (~1–2 seconds)
4. Speak the explanation aloud via macOS TTS
5. Resume playback

---

## Cost

API calls only happen when you trigger an explanation:

| Call | Cost |
|------|------|
| Whisper — 2 min of audio | ~$0.012 |
| GPT-4o — explanation | ~$0.01 |
| **Per explanation** | **~$0.02** |

50 explanations ≈ $1. Zero cost while passively listening.

---

## Fully Offline Option

Replace Whisper API with local transcription:

```bash
pip install faster-whisper
```

In `app.py`, change:
```python
self.transcriber = Transcriber()
```
to:
```python
from transcriber import LocalTranscriber
self.transcriber = LocalTranscriber(model_size="base.en")
```

The `base.en` model is ~140MB and runs on CPU in ~5–10 seconds for 2 minutes of audio. Use `small.en` for better accuracy, `tiny.en` for speed.

---

## Project Structure

```
podcast-copilot/
├── app.py              # Menu bar app, orchestrates everything
├── audio_buffer.py     # Thread-safe rolling RAM buffer
├── audio_capture.py    # System audio capture via BlackHole
├── wake_word.py        # Porcupine + fallback energy detector
├── transcriber.py      # Whisper API (+ local faster-whisper option)
├── explainer.py        # GPT-4o explanation
└── requirements.txt
```

---

## Troubleshooting

**No audio captured / empty transcripts**
→ Make sure BlackHole is set as part of a Multi-Output Device and that Multi-Output Device is selected as your Mac sound output in System Settings → Sound.

**Wake word not triggering**
→ If using Porcupine, check that PORCUPINE_ACCESS_KEY and PORCUPINE_MODEL_PATH are set correctly. If using fallback, speak clearly for ~1.5 seconds.

**Microphone permission denied**
→ System Settings → Privacy & Security → Microphone → enable Terminal (or your Python app).

**Media doesn't pause/resume**
→ The app sends the F8 media key via AppleScript. Check System Settings → Privacy & Security → Accessibility → allow Terminal. Falls back to direct Spotify AppleScript if F8 fails.

**Explanation is about the wrong thing**
→ Whisper is transcribing everything including ads. The GPT-4o prompt focuses on the most recent topic — if you triggered mid-ad, try again a few seconds into the content.
