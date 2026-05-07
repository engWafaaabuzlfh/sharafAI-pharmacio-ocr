"""Send a PDF to Gemini and ask for structured JSON extraction."""

from __future__ import annotations

import argparse
import os
from typing import Any

from utils.env import get_env_value
from utils.files import require_file
from utils.json_io import dumps_json, extract_json_object, write_json


DEFAULT_MODEL = "gemini-2.5-flash"

DEFAULT_PROMPT = """
You are an OCR and document-understanding engine for pharmacy invoices and tabular PDFs.
Return only valid JSON. Do not wrap the response in markdown.

Extract:
- document_type
- language
- vendor/supplier fields when available
- invoice number, date, tax, totals when available
- all tables as rows and columns
- line_items when the document looks like an invoice

Use this JSON shape:
{
  "document_type": "invoice|table|report|unknown",
  "language": "ar|en|mixed|unknown",
  "metadata": {},
  "tables": [
    {
      "page": 1,
      "title": null,
      "headers": [],
      "rows": [
        {"Col_1": "", "Col_2": ""}
      ]
    }
  ],
  "line_items": [],
  "raw_text": "",
  "warnings": []
}

If a value is missing, use null or an empty list. Preserve Arabic text as Arabic.
"""


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

    api_key = get_env_value("GOOGLE_API_KEY", "GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY in the environment or .env")

    pdf = require_file(pdf_path, label="PDF")

    client = genai.Client(api_key=api_key)
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract structured JSON from a PDF using Gemini.")
    parser.add_argument("pdf", help="Path to a PDF file")
    parser.add_argument("-o", "--output", default=None, help="Optional output JSON path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Gemini model, default: {DEFAULT_MODEL}")
    args = parser.parse_args()

    data = extract_pdf_to_json(args.pdf, output_path=args.output, model=args.model)
    print(dumps_json(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
