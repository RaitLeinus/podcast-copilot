"""
wake_word.py - Re-exports both wake word detector implementations.

  PorcupineWakeWordDetector  → wake_word_porcupine.py  (recommended)
  FallbackWakeWordDetector   → wake_word_fallback.py   (no setup needed)
"""

from wake_word_porcupine import PorcupineWakeWordDetector
from wake_word_fallback import FallbackWakeWordDetector

__all__ = ["PorcupineWakeWordDetector", "FallbackWakeWordDetector"]
