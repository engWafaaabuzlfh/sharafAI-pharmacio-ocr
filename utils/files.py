"""File path helpers shared by OCR runners."""

from __future__ import annotations

import os
from pathlib import Path


def require_file(path: str | os.PathLike[str], *, label: str = "File") -> Path:
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved
