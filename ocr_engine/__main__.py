"""Run: python -m ocr_engine (from pharmacio-ai directory, PYTHONPATH=. or pip install -e .)."""

import os

import uvicorn

if __name__ == "__main__":
    host = os.getenv("OCR_ENGINE_HOST", "0.0.0.0")
    port = int(os.getenv("OCR_ENGINE_PORT", "8080"))
    uvicorn.run("ocr_engine.main:app", host=host, port=port, reload=False)
