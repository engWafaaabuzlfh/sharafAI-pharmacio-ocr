"""Fetch uploaded PDF by storage key — S3 or local MEDIA_ROOT — without touching OCR code."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from ocr_engine.config import Settings

logger = logging.getLogger(__name__)


class FileFetchError(Exception):
    pass


def _safe_local_path(file_reference: str, settings: Settings) -> Path:
    root = Path(settings.local_media_root).resolve()
    raw = Path(file_reference)
    if raw.is_absolute():
        raise FileFetchError("Absolute file_reference is not allowed")
    # Normalize key as relative path segments (backend stores key as uploaded path)
    rel = Path(*[p for p in raw.parts if p not in ("..", ".")])
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as e:
        raise FileFetchError("Invalid file_reference path") from e
    return candidate


def fetch_file_to_path(*, file_reference: str, dest_path: str, settings: Settings) -> None:
    backend = (settings.storage_backend or "local").lower()
    if backend == "s3":
        _fetch_s3(file_reference, dest_path, settings)
    else:
        _fetch_local(file_reference, dest_path, settings)


def _fetch_local(file_reference: str, dest_path: str, settings: Settings) -> None:
    src = _safe_local_path(file_reference, settings)
    if not src.is_file():
        raise FileFetchError(f"File not found: {src}")
    Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest_path)
    logger.info("Copied local file %s -> %s", src, dest_path)


def _fetch_s3(key: str, dest_path: str, settings: Settings) -> None:
    if not settings.aws_storage_bucket_name:
        raise FileFetchError("AWS_STORAGE_BUCKET_NAME is not set for S3 storage")

    client = boto3.client(
        "s3",
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        endpoint_url=settings.aws_s3_endpoint_url,
        region_name=settings.aws_s3_region_name,
        config=Config(signature_version="s3v4"),
    )
    Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        client.download_file(settings.aws_storage_bucket_name, key, dest_path)
    except (ClientError, BotoCoreError) as e:
        raise FileFetchError(f"S3 download failed: {e}") from e
    logger.info("Downloaded s3://%s/%s -> %s", settings.aws_storage_bucket_name, key, dest_path)
