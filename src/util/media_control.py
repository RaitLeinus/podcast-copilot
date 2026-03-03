"""
media_control.py - Pause/resume media playback via F8 media key.

Works for Spotify, YouTube, Chrome, etc. via System Events.
Falls back to Spotify AppleScript if Accessibility permission is missing.
"""

import subprocess


def control_media(action: str):
    """Pause or resume media. action: 'pause' or 'play'."""
    script = 'tell application "System Events" to key code 16'
    try:
        subprocess.run(["osascript", "-e", script], check=True, capture_output=True)
    except Exception:
        applescript = (
            'tell application "Spotify" to pause'
            if action == "pause"
            else 'tell application "Spotify" to play'
        )
        try:
            subprocess.run(["osascript", "-e", applescript], capture_output=True)
        except Exception:
            pass
