"""Qwen2-VL model loading and generation helpers."""

from __future__ import annotations

from typing import Any

from PIL import Image

from services.huggingface.prompts import DEFAULT_MODEL, PAGE_PROMPT
from common.json_io import extract_json_object


def load_model(model_name: str = DEFAULT_MODEL):
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


def _generate_json_from_messages(
    messages: list[dict[str, Any]],
    *,
    model: Any,
    processor: Any,
    torch: Any,
    max_new_tokens: int,
) -> dict[str, Any]:
    from qwen_vl_utils import process_vision_info

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
    return extract_json_object(output_text)


def extract_page_json(
    image: Image.Image,
    *,
    page_number: int,
    model: Any,
    processor: Any,
    torch: Any,
    prompt: str = PAGE_PROMPT,
    max_new_tokens: int,
) -> dict[str, Any]:
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
    page_data = _generate_json_from_messages(
        messages,
        model=model,
        processor=processor,
        torch=torch,
        max_new_tokens=max_new_tokens,
    )
    page_data.setdefault("page", page_number)
    return page_data


def extract_document_json(
    images: list[Image.Image],
    *,
    model: Any,
    processor: Any,
    torch: Any,
    prompt: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {"type": "image", "image": image.convert("RGB")} for image in images
    ]
    document_prompt = (
        f"{prompt.strip()}\n\n"
        f"The document has {len(images)} page images, provided in order. "
        "Extract the whole document as one JSON object. Keep page numbers in page-level data."
    )
    content.append({"type": "text", "text": document_prompt})

    data = _generate_json_from_messages(
        [{"role": "user", "content": content}],
        model=model,
        processor=processor,
        torch=torch,
        max_new_tokens=max_new_tokens,
    )
    data.setdefault("pages", [])
    return data
