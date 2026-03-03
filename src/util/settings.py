"""
settings.py - Persistent environment variable storage in ~/.podcast_copilot_env.
"""

import os


def save_env(key: str, value: str):
    """Persist key=value to ~/.podcast_copilot_env"""
    env_file = os.path.expanduser("~/.podcast_copilot_env")
    lines = []
    found = False
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                if line.startswith(f"{key}="):
                    lines.append(f"{key}={value}\n")
                    found = True
                else:
                    lines.append(line)
    if not found:
        lines.append(f"{key}={value}\n")
    with open(env_file, "w") as f:
        f.writelines(lines)


def load_env():
    """Load key=value pairs from ~/.podcast_copilot_env into os.environ."""
    env_file = os.path.expanduser("~/.podcast_copilot_env")
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip())
