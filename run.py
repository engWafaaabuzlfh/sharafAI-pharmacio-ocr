"""Run the Pharmacio OCR FastAPI service."""

from __future__ import annotations

import os
import uvicorn


def main() -> None:
    host = os.getenv("OCR_ENGINE_HOST", "0.0.0.0")
    port = int(os.getenv("OCR_ENGINE_PORT", "8080"))
    uvicorn.run("ocr_engine.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
