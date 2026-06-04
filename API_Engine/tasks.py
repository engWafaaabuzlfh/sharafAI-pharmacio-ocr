"""Background job: resolve file -> run existing pipeline -> POST result to Django callback."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import uuid
from typing import Any

import requests

from API_Engine.config import Settings, get_settings
from API_Engine.storage import FileFetchError, fetch_file_to_path
from services.ocr_engines import extract_pdf_to_json, normalize_engine_name

logger = logging.getLogger(__name__)


def _item_value(item: object, key: str) -> str:
    if isinstance(item, dict):
        return item.get(key, "")
    return getattr(item, key, "")


def _prepare_callback_items(target_items: list[object] | None) -> list[dict[str, object]]:
    if not target_items:
        return []

    results = []
    for item in target_items:
        results.append(
            {
                "drug_name": _item_value(item, "product_name"),
                "company": "",
                "price": 0.0,
                "confidence": 0.0,
                "review_required": False,
            }
        )
    return results


def run_ocr_job(
    job_id: str,
    file_reference: str,
    settings: Settings | None = None,
    ocr_engine: str = "easyocr",
    target_items: list[dict[str, str]] | None = None,
) -> None:
    """
    Download PDF by storage key, run integrated_pipeline.process_pdf_to_json unchanged,
    POST {job_id, payload} to backend /api/v1/ocr/result/.
    """
    settings = settings or get_settings()
    callback_url = f"{settings.backend_base_url.rstrip('/')}/api/v1/ocr/result/"

    with tempfile.TemporaryDirectory(prefix="ocr_engine_") as tmp_root:
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
        normalized_engine = normalize_engine_name(ocr_engine)
        try:
            merged_data = extract_pdf_to_json(
                normalized_engine,
                local_pdf,
                output_path=work_dir if normalized_engine == "easyocr" else os.path.join(work_dir, f"{normalized_engine}_result.json"),
            )
        except Exception:
            logger.exception("Job %s: OCR engine %s failed", job_id, ocr_engine)
            return

        if not merged_data:
            logger.error("Job %s: pipeline returned no merged_data", job_id)
            return

    headers = {"Authorization": settings.internal_service_token}
    body = {
        "job_id": job_id,
        "payload": {
            "items": _prepare_callback_items(target_items),
        },
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
