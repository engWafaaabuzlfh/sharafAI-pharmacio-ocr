"""EasyOCR + OpenCV pipeline service."""

from services.easyocr_pipeline.pipeline import process_pdf_to_json

__all__ = ["process_pdf_to_json"]
