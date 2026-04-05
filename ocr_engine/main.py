"""
FastAPI OCR engine — encapsulates integrated_pipeline for pharmacio-backend dispatch.

Backend sends: POST JSON { job_id, file_reference }, Authorization: AI_ENGINE_API_KEY
This service responds 202 immediately, then runs process_pdf_to_json and POSTs to
/api/v1/ocr/result/ with Authorization: INTERNAL_SERVICE_TOKEN.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from ocr_engine.config import get_settings
from ocr_engine.tasks import run_ocr_job

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Pharmacio OCR Engine", version="1.0.0")


class ProcessRequest(BaseModel):
    job_id: str = Field(..., description="OCR job UUID string from backend")
    file_reference: str = Field(..., description="Storage key (S3 key or local media path)")


def _check_dispatch_auth(authorization: str | None) -> None:
    settings = get_settings()
    expected = settings.ai_engine_api_key
    if not expected:
        logger.error("AI_ENGINE_API_KEY is not configured")
        raise HTTPException(status_code=503, detail="OCR engine not configured")
    if (authorization or "") != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _enqueue_process(
    body: ProcessRequest,
    background_tasks: BackgroundTasks,
    authorization: str | None,
) -> dict[str, str]:
    _check_dispatch_auth(authorization)
    settings = get_settings()
    background_tasks.add_task(run_ocr_job, body.job_id, body.file_reference, settings)
    logger.info("Accepted job_id=%s file_reference=%s", body.job_id, body.file_reference)
    return {"status": "accepted"}


@app.post("/v1/process", status_code=202)
def accept_process(
    body: ProcessRequest,
    background_tasks: BackgroundTasks,
    authorization: Annotated[str | None, Header()] = None,
) -> dict[str, str]:
    """
    Accept OCR job (matches typical OCR_ENGINE_PROCESS_URL .../v1/process).
    Returns before pipeline completes so Celery dispatch stays within timeout.
    """
    return _enqueue_process(body, background_tasks, authorization)


@app.post("/process", status_code=202, include_in_schema=False)
def accept_process_alias(
    body: ProcessRequest,
    background_tasks: BackgroundTasks,
    authorization: Annotated[str | None, Header()] = None,
) -> dict[str, str]:
    """Optional path without /v1 prefix if OCR_ENGINE_PROCESS_URL omits it."""
    return _enqueue_process(body, background_tasks, authorization)
