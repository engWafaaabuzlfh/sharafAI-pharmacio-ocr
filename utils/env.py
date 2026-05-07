"""Small environment helpers used by OCR engines."""

from __future__ import annotations

import os
from pathlib import Path


def read_dotenv_value(key: str, env_path: str | os.PathLike[str] | None = None) -> str | None:
    path = Path(env_path) if env_path else Path.cwd() / ".env"
    if not path.is_file():
        return None

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == key:
            return value.strip().strip('"').strip("'") or None
    return None


def get_env_value(*keys: str) -> str | None:
    for key in keys:
        value = os.getenv(key) or read_dotenv_value(key)
        if value:
            return value
    return None
