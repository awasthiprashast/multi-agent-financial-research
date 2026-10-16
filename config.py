"""
config.py — load settings from a local `.env` file.

Real environment variables always win over the file, so deployments can inject
secrets without a file at all. The file is parsed by hand (KEY=value, `#` comments,
optional quotes) to avoid another dependency. `.env` is git-ignored; `.env.example`
documents every setting.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ChromaDB phones home by default; this is a research tool, not telemetry.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


def load_env(path: Path | None = None) -> None:
    path = path or ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value:
            os.environ.setdefault(key, value)


load_env()
