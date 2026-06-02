"""Command-line entrypoint for Hugging Face Qwen-VL OCR."""

from __future__ import annotations

import argparse

from services.huggingface.prompts import DEFAULT_MODEL
from services.huggingface.runner import extract_pdf_to_json
from common.json_io import dumps_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract structured JSON from a PDF using Qwen2-VL.")
    parser.add_argument("pdf", help="Path to a PDF file")
    parser.add_argument("-o", "--output", default=None, help="Optional output JSON path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"HF model, default: {DEFAULT_MODEL}")
    parser.add_argument("--dpi", type=int, default=180, help="PDF render DPI, default: 180")
    parser.add_argument("--max-new-tokens", type=int, default=4096, help="Generation budget")
    parser.add_argument(
        "--mode",
        choices=["document", "pages"],
        default="document",
        help="document sends all page images in one request; pages processes one page at a time",
    )
    args = parser.parse_args()

    data = extract_pdf_to_json(
        args.pdf,
        output_path=args.output,
        model_name=args.model,
        dpi=args.dpi,
        max_new_tokens=args.max_new_tokens,
        mode=args.mode,
    )
    print(dumps_json(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
