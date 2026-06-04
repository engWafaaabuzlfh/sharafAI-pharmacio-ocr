"""High-level Hugging Face Qwen-VL extraction runner."""

from __future__ import annotations

import os
from typing import Any

from PIL import Image

from services.huggingface.model import extract_document_json, extract_page_json, load_model
from common.pdf import render_pdf_pages
from services.huggingface.prompts import DEFAULT_MODEL, DEFAULT_PROMPT, PAGE_PROMPT
from common.files import require_file
from common.json_io import write_json

print("Hugging Face Qwen-VL OCR engine loaded with model:", DEFAULT_MODEL)
print("Default prompt:", DEFAULT_PROMPT)
def extract_images_to_json(
    images: list[Image.Image],
    *,
    model_name: str = DEFAULT_MODEL,
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = 2048,
    mode: str = "document",
) -> dict[str, Any]:
    model, processor, torch = load_model(model_name)

    if mode == "document":
        print(f"Processing {len(images)} page(s) in one Qwen2-VL request with {model_name}")
        data = extract_document_json(
            images,
            model=model,
            processor=processor,
            torch=torch,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        return {
            "engine": "huggingface-qwen-vl",
            "model": model_name,
            "mode": mode,
            "result": data,
        }

    if mode != "pages":
        raise ValueError("mode must be 'document' or 'pages'")

    pages = []
    for page_number, image in enumerate(images, start=1):
        print(f"Processing page {page_number}/{len(images)} with {model_name}")
        pages.append(
            extract_page_json(
                image,
                page_number=page_number,
                model=model,
                processor=processor,
                torch=torch,
                prompt=PAGE_PROMPT,
                max_new_tokens=max_new_tokens,
            )
        )

    return {
        "engine": "huggingface-qwen-vl",
        "model": model_name,
        "mode": mode,
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
    mode: str = "document",
) -> dict[str, Any]:
    pdf = require_file(pdf_path, label="PDF")
    images = render_pdf_pages(pdf, dpi=dpi)
    data = extract_images_to_json(
        images,
        model_name=model_name,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        mode=mode,
    )

    if output_path:
        write_json(data, output_path)
    return data
