"""Gemini PDF extraction client."""

from __future__ import annotations

import os
from typing import Any

from services.gemini.prompts import DEFAULT_MODEL, DEFAULT_PROMPT
from common.env import get_env_value
from common.files import require_file
from common.json_io import extract_json_object, write_json


def _get_api_key() -> str:
    api_key = get_env_value("GOOGLE_API_KEY", "GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY in the environment or .env")
    return api_key


def extract_pdf_to_json(
    pdf_path: str | os.PathLike[str],
    *,
    output_path: str | os.PathLike[str] | None = None,
    model: str = DEFAULT_MODEL,
    prompt: str = DEFAULT_PROMPT,
) -> dict[str, Any]:
    """Upload a PDF to Gemini and return structured JSON."""
    from google import genai
    from google.genai import types

    pdf = require_file(pdf_path, label="PDF")
    client = genai.Client(api_key=_get_api_key())
    uploaded_file = client.files.upload(file=str(pdf))
    response = client.models.generate_content(
        model=model,
        contents=[uploaded_file, prompt],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )

    data = extract_json_object(response.text or "{}")
    if output_path:
        write_json(data, output_path)
    return data
