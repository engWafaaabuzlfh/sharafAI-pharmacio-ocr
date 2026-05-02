"""Runtime-heavy resources (loaded lazily)."""

from __future__ import annotations

import os
import warnings

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_reader = None


def get_reader():
    global _reader
    if _reader is None:
        import easyocr

        _reader = easyocr.Reader(["ar", "en"], gpu=False)
    return _reader
