"""Unified OCR smoke-test helper.

Examples:
    python test_ocr_image.py easyocr media/invoices/test.jpg
    python test_ocr_image.py gemini media/invoices/test.pdf -o gemini_result.json
    python test_ocr_image.py huggingface media/invoices/test.pdf -o hf_qwen_result.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _write_or_print(data: dict[str, Any], output_path: str | None) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=2)
    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        print(f"Saved JSON: {output}")
    print(text)


def run_easyocr(path: Path) -> dict[str, Any]:
    from ocr_pipeline.ocr_extraction import extract_arabic_from_cell_image

    text, details = extract_arabic_from_cell_image(str(path))
    return {
        "engine": "easyocr",
        "input": str(path),
        "text": text,
        "details": details,
    }


def run_gemini(path: Path, output_path: str | None, model: str | None) -> dict[str, Any]:
    from ocr_gemini.extractor import DEFAULT_MODEL, extract_pdf_to_json

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
) -> dict[str, Any]:
    from ocr_huggingFace.extractor import DEFAULT_MODEL, extract_pdf_to_json

    return extract_pdf_to_json(
        path,
        output_path=output_path,
        model_name=model or DEFAULT_MODEL,
        dpi=dpi,
        max_new_tokens=max_new_tokens,
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
        default=2048,
        help="Generation budget per page for Hugging Face",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"Input file not found: {input_path}")
        return 1

    if args.engine == "easyocr":
        data = run_easyocr(input_path)
        _write_or_print(data, args.output)
    elif args.engine == "gemini":
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
        )
        if not args.output:
            _write_or_print(data, None)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
