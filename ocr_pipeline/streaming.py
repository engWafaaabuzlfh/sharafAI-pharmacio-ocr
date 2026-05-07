"""Page-by-page OCR pipeline orchestration."""

from __future__ import annotations

import glob
import json
import os

import cv2

from ocr_pipeline.cell_detection import (
    _fallback_grid_detection,
    detect_cells,
    preprocess_image,
    save_cell_positions,
)
from ocr_pipeline.merge import read_and_merge_json_files_simple
from ocr_pipeline.ocr_extraction import process_cells_with_positions
from ocr_pipeline.preprocessing import process_image_normalize_only
from ocr_pipeline.settings import EXPERIMENTAL_CONFIGS


def _read_dotenv_value(key: str) -> str | None:
    env_path = os.path.join(os.getcwd(), ".env")
    if not os.path.isfile(env_path):
        return None

    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() == key:
                return value.strip().strip('"').strip("'") or None
    return None


def _get_poppler_path() -> str | None:
    return os.getenv("POPPLER_PATH") or _read_dotenv_value("POPPLER_PATH")


def iter_pdf_images(pdf_path: str, output_dir: str, dpi: int = 300):
    """Convert and yield one PDF page image at a time."""
    from pdf2image import convert_from_path, pdfinfo_from_path

    raw_images_dir = os.path.join(output_dir, "seperate_image")
    os.makedirs(raw_images_dir, exist_ok=True)
    poppler_path = _get_poppler_path()

    try:
        info = pdfinfo_from_path(pdf_path, poppler_path=poppler_path)
        total_pages = int(info.get("Pages", 0))
        print(f"Number of extracted images: {total_pages}")

        for page_num in range(1, total_pages + 1):
            images = convert_from_path(
                pdf_path,
                dpi=dpi,
                first_page=page_num,
                last_page=page_num,
                poppler_path=poppler_path,
            )
            if not images:
                continue

            filename = f"page_{page_num:03d}_raw.jpg"
            filepath = os.path.join(raw_images_dir, filename)
            images[0].save(filepath, "JPEG", quality=95, optimize=True)
            yield page_num, filepath
    except Exception as e:
        print(f"Error converting PDF page-by-page: {e}")


def _page_steps_dir(processed_dir: str, image_path: str) -> str:
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(processed_dir, f"{image_name}_steps")


def process_normalized_image(img_path: str, exp_dir: str):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Cannot read image: {img_path}")
        return None

    config = EXPERIMENTAL_CONFIGS[0]
    cells_dir = os.path.join(exp_dir, "cells")
    debug_dir = os.path.join(exp_dir, "debug")
    os.makedirs(cells_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    binary, gray, enhanced = preprocess_image(img, config)
    cv2.imwrite(os.path.join(debug_dir, "enhanced.jpg"), enhanced)
    cv2.imwrite(os.path.join(debug_dir, "binary.jpg"), binary)

    cells = detect_cells(binary, gray.shape, config)
    if len(cells) < 5:
        cells = _fallback_grid_detection(binary, gray.shape, config)

    debug_img = img.copy()
    for x, y, w, h in cells:
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv2.imwrite(os.path.join(debug_dir, "detected_cells.jpg"), debug_img)

    cells = sorted(cells, key=lambda b: (b[1], b[0]))
    for i, (x, y, w, h) in enumerate(cells):
        margin = config["cell_margin"]
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(gray.shape[1], x + w + margin)
        y2 = min(gray.shape[0], y + h + margin)
        cell_img = enhanced[y1:y2, x1:x2]
        cv2.imwrite(os.path.join(cells_dir, f"cell_{i:04d}.jpg"), cell_img)

    save_cell_positions(cells, exp_dir)
    print(f"Processed and saved: {img_path}")
    return exp_dir


def _write_final_json(output_base_dir: str, merged_data: dict):
    output_json = os.path.join(output_base_dir, "final_extracted_text.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    return output_json


def process_pdf_to_json(pdf_path: str, output_base_dir: str | None = None, dpi: int = 300):
    print("=" * 70)
    print("Starting PDF processing and text extraction")
    print("=" * 70)

    if output_base_dir is None:
        pdf_dir = os.path.dirname(pdf_path)
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_base_dir = os.path.join(pdf_dir, f"{pdf_name}_output")

    output_base_dir = os.path.abspath(output_base_dir)
    print(f"\nOutput folder: {output_base_dir}")
    os.makedirs(output_base_dir, exist_ok=True)

    processed_dir = os.path.join(output_base_dir, "processed")
    experiment_dir = os.path.join(output_base_dir, "experiment_results")
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(experiment_dir, exist_ok=True)

    print("\nStreaming pages: convert -> normalize -> detect cells -> OCR")
    processed_count = 0
    converted_count = 0

    for page_num, image_path in iter_pdf_images(pdf_path, output_base_dir, dpi):
        converted_count += 1
        page_name = os.path.splitext(os.path.basename(image_path))[0]
        page_folder = f"{page_name}_steps"
        page_exp_dir = os.path.join(experiment_dir, page_folder)

        print(f"\nPage {page_num}: Processing {os.path.basename(image_path)}")
        _, steps = process_image_normalize_only(image_path, processed_dir)
        if not steps:
            print(f"Page {page_num}: image normalization failed")
            continue

        normalized_path = os.path.join(_page_steps_dir(processed_dir, image_path), "02_normalized.png")
        if not os.path.exists(normalized_path):
            print(f"Page {page_num}: normalized image was not created")
            continue

        if not process_normalized_image(normalized_path, page_exp_dir):
            print(f"Page {page_num}: cell detection failed")
            continue

        result = process_cells_with_positions(page_exp_dir, experiment_dir)
        if result:
            processed_count += 1
            print(f"Page {page_num}: OCR completed")

    if converted_count == 0:
        print("Failed to convert PDF")
        return None, None

    print(f"\nText extraction completed. Processed {processed_count}/{converted_count} pages")
    print("\nMerging page results into JSON")

    json_pattern = os.path.join(experiment_dir, "*", "easyocr_results", "all_results.json")
    json_pattern = json_pattern.replace("\\", "/")
    json_files = glob.glob(json_pattern, recursive=True)
    print(f"Found {len(json_files)} JSON files")

    if not json_files:
        print("No JSON files found anywhere")
        return None, None

    merged_data = read_and_merge_json_files_simple(json_pattern)
    if not merged_data:
        print("\nFailed to get merged data")
        return None, None

    output_json = _write_final_json(output_base_dir, merged_data)

    print("Processing completed successfully!")
    print(f"Final JSON file: {output_json}")
    print(f"Number of pages: {len(merged_data)}")

    if merged_data:
        print("\nSample of extracted data:")
        for page_name, page_data in list(merged_data.items())[:2]:
            print(f"\n  {page_name}: {len(page_data)} rows")
            if page_data:
                print(f"  First row: {page_data[0]}")

    return output_json, merged_data
