"""Background job: resolve file -> run existing pipeline -> POST result to Django callback."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import uuid
from typing import Any

import requests

from ocr_engine.config import Settings, get_settings
from ocr_engine.storage import FileFetchError, fetch_file_to_path

logger = logging.getLogger(__name__)


def _normalize_engine_name(ocr_engine: str | None) -> str:
    value = (ocr_engine or "easyocr").strip().lower().replace("-", "_")
    aliases = {
        "easy": "easyocr",
        "easy_ocr": "easyocr",
        "hf": "huggingface",
        "hugging_face": "huggingface",
        "qwen": "huggingface",
        "qwen2_vl": "huggingface",
        "google": "gemini",
    }
    return aliases.get(value, value)


def _run_selected_engine(local_pdf: str, work_dir: str, ocr_engine: str) -> dict[str, Any] | None:
    engine = _normalize_engine_name(ocr_engine)
    if engine == "easyocr":
        from services.easyocr_pipeline.pipeline import process_pdf_to_json

        _, merged_data = process_pdf_to_json(local_pdf, output_base_dir=work_dir)
        return merged_data

    if engine == "gemini":
        from services.gemini import extract_pdf_to_json

        return extract_pdf_to_json(local_pdf, output_path=os.path.join(work_dir, "gemini_result.json"))

    if engine == "huggingface":
        from services.huggingface import extract_pdf_to_json

        return extract_pdf_to_json(local_pdf, output_path=os.path.join(work_dir, "huggingface_result.json"))

    raise ValueError(f"Unsupported OCR engine: {ocr_engine}")


def run_ocr_job(
    job_id: str,
    file_reference: str,
    settings: Settings | None = None,
    ocr_engine: str = "easyocr",
) -> None:
    """
    Download PDF by storage key, run integrated_pipeline.process_pdf_to_json unchanged,
    POST {job_id, payload} to backend /api/v1/ocr/result/.
    """
    settings = settings or get_settings()
    callback_url = f"{settings.backend_base_url.rstrip('/')}/api/v1/ocr/result/"

    tmp_root = tempfile.mkdtemp(prefix="ocr_engine_")
    pdf_name = f"{uuid.uuid4().hex}.pdf"
    local_pdf = os.path.join(tmp_root, pdf_name)
    work_dir = os.path.join(tmp_root, "work")

    try:
        fetch_file_to_path(file_reference=file_reference, dest_path=local_pdf, settings=settings)
    except FileFetchError as e:
        logger.error("Job %s: could not fetch file_reference=%s: %s", job_id, file_reference, e)
        return
    except Exception:
        logger.exception("Job %s: unexpected fetch error for %s", job_id, file_reference)
        return

    merged_data: dict[str, Any] | None = None
    try:
        # Lazy import inside dispatcher so optional engines load only when selected.
        merged_data = _run_selected_engine(local_pdf, work_dir, ocr_engine)
    except Exception:
        logger.exception("Job %s: OCR engine %s failed", job_id, ocr_engine)
        return

    if not merged_data:
        logger.error("Job %s: pipeline returned no merged_data", job_id)
        return

    headers = {"Authorization": settings.internal_service_token}
    body = {
        "job_id": job_id,
        "ocr_engine": _normalize_engine_name(ocr_engine),
        "payload": merged_data,
    }
    try:
        r = requests.post(
            callback_url,
            json=body,
            headers=headers,
            timeout=settings.callback_timeout_seconds,
        )
        r.raise_for_status()
        logger.info("Job %s: callback OK (%s)", job_id, r.status_code)
    except requests.RequestException as e:
        resp_text = None
        if getattr(e, "response", None) is not None:
            resp_text = e.response.text
        logger.error("Job %s: callback failed: %s body=%s", job_id, e, resp_text)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
