"""Send a PDF to Gemini and ask for structured JSON extraction."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


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


def _read_dotenv_value(key: str) -> str | None:
    env_path = Path.cwd() / ".env"
    if not env_path.is_file():
        return None

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == key:
            return value.strip().strip('"').strip("'") or None
    return None


def _json_from_text(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    return json.loads(cleaned)


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

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or _read_dotenv_value("GOOGLE_API_KEY") or _read_dotenv_value("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY in the environment or .env")

    pdf = Path(pdf_path)
    if not pdf.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf}")

    client = genai.Client(api_key=api_key)
    uploaded_file = client.files.upload(file=str(pdf))
    response = client.models.generate_content(
        model=model,
        contents=[uploaded_file, prompt],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )

    data = _json_from_text(response.text or "{}")
    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract structured JSON from a PDF using Gemini.")
    parser.add_argument("pdf", help="Path to a PDF file")
    parser.add_argument("-o", "--output", default=None, help="Optional output JSON path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Gemini model, default: {DEFAULT_MODEL}")
    args = parser.parse_args()

    data = extract_pdf_to_json(args.pdf, output_path=args.output, model=args.model)
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
