# 🎙 Podcast Copilot

A Mac menu bar app that listens to whatever you're playing in Spotify and explains content on voice command. Say **"explain"** (optionally followed by a topic like "explain the inflation rate") and it pauses, explains what you just heard, then resumes.

## How It Works

```
System audio (Spotify)
        ↓
  ScreenCaptureKit (macOS 13+)
        ↓
  Rolling 2-min audio buffer  ← stored in RAM only, nothing sent anywhere
        ↓ (only when wake word fires)
  Whisper API  ←  transcribes the buffered podcast audio
        ↓
  GPT-4o audio  ←  generates + speaks explanation in one streaming pass
        ↓
  Resume playback


Microphone → Porcupine (on-device) → wake word "explain"
             (no audio leaves your machine for wake word detection)

After wake word fires:
  Microphone → Whisper API  ←  transcribes your optional follow-up topic
```

**Privacy model:** The podcast audio never leaves your computer until you explicitly ask for an explanation. The wake word runs 100% on-device. Only when you say "explain" does ~30 seconds of buffered audio get sent to Whisper.

---

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Grant Screen Recording permission

ScreenCaptureKit captures system audio directly — no virtual audio device needed.

System Settings → Privacy & Security → Screen Recording → enable your terminal app.

### 3. Set up Porcupine wake word (recommended)

Porcupine runs fully on-device with ~50ms latency. A bundled wake word model (`explain_en_mac_v4_0_0.ppn`) is included — you just need a free API key to activate it.

1. Free account at https://console.picovoice.ai/ → copy your **Access Key**
2. Set the environment variable:
   ```bash
   export PORCUPINE_ACCESS_KEY=your-access-key
   ```

That's it — the bundled "explain" wake word model is picked up automatically.

> **Custom wake word?** Create one in the Picovoice Console, download the `.ppn` for Mac, and set `PORCUPINE_MODEL_PATH=/path/to/your_model.ppn`.

> **No Porcupine?** The app falls back to energy-based detection — it triggers when you speak for ~1.5 seconds. Works fine in quiet environments but has no phrase recognition.

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

**To get an explanation:** say **"explain"** (wait for the chime, then optionally add context). The app will:
1. Immediately pause playback
2. Play a chime and listen for your follow-up (e.g. "the NATO founding" or "who is Milton Friedman")
3. Send buffered audio + your command to Whisper in parallel
4. Stream the explanation as audio directly from GPT-4o (starts speaking within ~1s)
5. Resume playback

---

## Cost

API calls only happen when you trigger an explanation:

| Call | Cost |
|------|------|
| Whisper — 2 min of podcast audio | ~$0.012 |
| Whisper — your voice command (~3s) | ~$0.001 |
| GPT-4o audio — explanation + speech | ~$0.01 |
| **Per explanation** | **~$0.02** |

50 explanations ≈ $1. Zero cost while passively listening.

---

## Project Structure

```
podcast-copilot/
├── app.py                        # Menu bar app, orchestrates everything
├── audio_buffer.py               # Thread-safe rolling RAM buffer
├── audio_capture.py              # System audio capture via ScreenCaptureKit
├── wake_word.py                  # Porcupine + fallback energy detector
├── transcriber.py                # Whisper API transcription
├── explainer.py                  # GPT-4o audio streaming explanation
├── explain_en_mac_v4_0_0.ppn     # Bundled Porcupine wake word model
└── requirements.txt
```

---

## Troubleshooting

**No audio captured / empty transcripts**
→ Make sure Screen Recording permission is granted for your terminal app in System Settings → Privacy & Security → Screen Recording.

**Wake word not triggering**
→ Check that `PORCUPINE_ACCESS_KEY` is set. If using fallback, speak clearly for ~1.5 seconds.

**Microphone permission denied**
→ System Settings → Privacy & Security → Microphone → enable Terminal (or your Python app).

**Media doesn't pause/resume**
→ The app sends the F8 media key via AppleScript. Check System Settings → Privacy & Security → Accessibility → allow Terminal. Falls back to direct Spotify AppleScript if F8 fails.

**Explanation is about the wrong thing**
→ Say "explain" followed by the specific topic you want after the chime (e.g. "explain the Marshall Plan"). This focuses GPT on that topic within the transcript context.
