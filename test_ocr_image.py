"""Unified OCR smoke-test helper.

Examples:
    python test_ocr_image.py easyocr media/invoices/test.jpg
    python test_ocr_image.py gemini media/invoices/test.pdf -o gemini_result.json
    python test_ocr_image.py huggingface media/invoices/test.pdf -o hf_qwen_result.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common.files import require_file
from common.json_io import dumps_json, write_json
from services.ocr_engines import normalize_engine_name


def _write_or_print(data: dict[str, Any], output_path: str | None) -> None:
    if output_path:
        output = write_json(data, output_path)
        print(f"Saved JSON: {output}")
    print(dumps_json(data))


def run_easyocr(path: Path) -> dict[str, Any]:
    from services.easyocr_pipeline.ocr_extraction import extract_arabic_from_cell_image

    text, details = extract_arabic_from_cell_image(str(path))
    return {
        "engine": "easyocr",
        "input": str(path),
        "text": text,
        "details": details,
    }


def run_gemini(path: Path, output_path: str | None, model: str | None) -> dict[str, Any]:
    from services.gemini import extract_pdf_to_json
    from services.gemini.prompts import DEFAULT_MODEL

    return extract_pdf_to_json(
        path,
        output_path=output_path,
        model=model or DEFAULT_MODEL,
    )


def run_huggingface(
    path: Path,
    output_path: str | None,
    model: str | None,
    dpi: int,
    max_new_tokens: int,
    mode: str,
) -> dict[str, Any]:
    from services.huggingface import extract_pdf_to_json
    from services.huggingface.prompts import DEFAULT_MODEL

    return extract_pdf_to_json(
        path,
        output_path=output_path,
        model_name=model or DEFAULT_MODEL,
        dpi=dpi,
        max_new_tokens=max_new_tokens,
        mode=mode,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Test one of the OCR engines.")
    parser.add_argument(
        "engine",
        choices=["easyocr", "gemini", "huggingface", "hf"],
        help="OCR engine to test",
    )
    parser.add_argument("input", help="Image path for easyocr, PDF path for gemini/huggingface")
    parser.add_argument("-o", "--output", default=None, help="Optional output JSON path")
    parser.add_argument("--model", default=None, help="Optional Gemini/Hugging Face model name")
    parser.add_argument("--dpi", type=int, default=180, help="PDF render DPI for Hugging Face")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Generation budget per page for Hugging Face",
    )
    parser.add_argument(
        "--mode",
        choices=["document", "pages"],
        default="document",
        help="Hugging Face mode: document sends all pages together, pages sends one page at a time",
    )
    args = parser.parse_args()

    try:
        input_path = require_file(args.input, label="Input file")
    except FileNotFoundError as e:
        print(e)
        return 1

    normalized_engine = normalize_engine_name(args.engine)
    if normalized_engine == "easyocr":
        data = run_easyocr(input_path)
        _write_or_print(data, args.output)
    elif normalized_engine == "gemini":
        data = run_gemini(input_path, args.output, args.model)
        if not args.output:
            _write_or_print(data, None)
    else:
        data = run_huggingface(
            input_path,
            args.output,
            args.model,
            args.dpi,
            args.max_new_tokens,
            args.mode,
        )
        if not args.output:
            _write_or_print(data, None)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
