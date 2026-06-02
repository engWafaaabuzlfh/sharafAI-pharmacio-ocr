"""Command-line entrypoint for Gemini OCR."""

from __future__ import annotations

import argparse

from services.gemini.client import extract_pdf_to_json
from services.gemini.prompts import DEFAULT_MODEL
from common.json_io import dumps_json


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
