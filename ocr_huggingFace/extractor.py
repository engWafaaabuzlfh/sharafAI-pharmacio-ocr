"""Local Hugging Face PDF-to-JSON extraction using Qwen2-VL."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image


DEFAULT_MODEL = "Qwen/Qwen2-VL-7B-Instruct"

DEFAULT_PROMPT = """
You are an OCR and document-understanding engine for pharmacy invoices, tables, and reports.
Analyze this page image and return only valid JSON. Do not wrap the response in markdown.

Extract all visible Arabic and English text. Preserve Arabic text as Arabic.
If the page has tables, reconstruct rows and columns as accurately as possible.

Use this JSON shape:
{
  "page": null,
  "document_type": "invoice|table|report|unknown",
  "language": "ar|en|mixed|unknown",
  "metadata": {},
  "tables": [
    {
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

If a value is missing, use null or an empty list.
"""


def _render_pdf_pages(pdf_path: Path, *, dpi: int = 180) -> list[Image.Image]:
    """Render PDF pages with PyMuPDF, avoiding the external Poppler dependency."""
    import fitz

    scale = dpi / 72
    matrix = fitz.Matrix(scale, scale)
    pages: list[Image.Image] = []

    with fitz.open(str(pdf_path)) as doc:
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pages.append(image)
    return pages


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
        return {"raw_response": text, "warnings": ["Model did not return valid JSON"]}


def _load_model(model_name: str):
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    return model, processor, torch


def _extract_page_json(
    image: Image.Image,
    *,
    page_number: int,
    model: Any,
    processor: Any,
    torch: Any,
    prompt: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    from qwen_vl_utils import process_vision_info

    page_prompt = f"{prompt.strip()}\n\nThis is page number {page_number}. Set the JSON page field to {page_number}."
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image.convert("RGB")},
                {"type": "text", "text": page_prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_ids_trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    page_data = _extract_json_object(output_text)
    page_data.setdefault("page", page_number)
    return page_data


def extract_images_to_json(
    images: list[Image.Image],
    *,
    model_name: str = DEFAULT_MODEL,
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = 2048,
) -> dict[str, Any]:
    model, processor, torch = _load_model(model_name)

    pages = []
    for page_number, image in enumerate(images, start=1):
        print(f"Processing page {page_number}/{len(images)} with {model_name}")
        pages.append(
            _extract_page_json(
                image,
                page_number=page_number,
                model=model,
                processor=processor,
                torch=torch,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )
        )

    return {
        "engine": "huggingface-qwen-vl",
        "model": model_name,
        "pages": pages,
    }


def extract_pdf_to_json(
    pdf_path: str | os.PathLike[str],
    *,
    output_path: str | os.PathLike[str] | None = None,
    model_name: str = DEFAULT_MODEL,
    prompt: str = DEFAULT_PROMPT,
    dpi: int = 180,
    max_new_tokens: int = 2048,
) -> dict[str, Any]:
    pdf = Path(pdf_path)
    if not pdf.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf}")

    images = _render_pdf_pages(pdf, dpi=dpi)
    data = extract_images_to_json(
        images,
        model_name=model_name,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )

    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract structured JSON from a PDF using Qwen2-VL.")
    parser.add_argument("pdf", help="Path to a PDF file")
    parser.add_argument("-o", "--output", default=None, help="Optional output JSON path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"HF model, default: {DEFAULT_MODEL}")
    parser.add_argument("--dpi", type=int, default=180, help="PDF render DPI, default: 180")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Generation budget per page")
    args = parser.parse_args()

    data = extract_pdf_to_json(
        args.pdf,
        output_path=args.output,
        model_name=args.model,
        dpi=args.dpi,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
