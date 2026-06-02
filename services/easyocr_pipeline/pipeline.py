"""Top-level pipeline orchestration."""

from services.easyocr_pipeline.streaming import process_pdf_to_json

__all__ = ["process_pdf_to_json"]
