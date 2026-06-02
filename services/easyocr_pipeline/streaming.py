"""Page-by-page OCR pipeline orchestration."""

from __future__ import annotations

import glob
import json
import logging
import os

import cv2

from services.easyocr_pipeline.cell_detection import process_normalized_image_data
from services.easyocr_pipeline.merge import read_and_merge_json_files_simple
from services.easyocr_pipeline.ocr_extraction import process_cells_with_positions
from services.easyocr_pipeline.preprocessing import _get_poppler_path, process_image_normalize_only
from services.easyocr_pipeline.settings import EXPERIMENTAL_CONFIGS

logger = logging.getLogger(__name__)


def iter_pdf_images(pdf_path: str, output_dir: str, dpi: int = 300):
    """Convert and yield one PDF page image at a time."""
    from pdf2image import convert_from_path, pdfinfo_from_path

    raw_images_dir = os.path.join(output_dir, "seperate_image")
    os.makedirs(raw_images_dir, exist_ok=True)
    poppler_path = _get_poppler_path()

    try:
        info = pdfinfo_from_path(pdf_path, poppler_path=poppler_path)
        total_pages = int(info.get("Pages", 0))
        logger.info("Number of extracted images: %s", total_pages)

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
    except Exception:
        logger.exception("Error converting PDF page-by-page: %s", pdf_path)


def _page_steps_dir(processed_dir: str, image_path: str) -> str:
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(processed_dir, f"{image_name}_steps")


def process_normalized_image(img_path: str, exp_dir: str):
    img = cv2.imread(img_path)
    if img is None:
        logger.error("Cannot read image: %s", img_path)
        return None

    config = EXPERIMENTAL_CONFIGS[0]
    cells = process_normalized_image_data(img, exp_dir, config)
    if not cells:
        logger.warning("No cells detected in normalized image: %s", img_path)
        return None

    logger.info("Processed and saved: %s", img_path)
    return exp_dir


def _write_final_json(output_base_dir: str, merged_data: dict):
    output_json = os.path.join(output_base_dir, "final_extracted_text.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    return output_json


def process_pdf_to_json(pdf_path: str, output_base_dir: str | None = None, dpi: int = 300):
    logger.info("Starting PDF processing and text extraction")

    if output_base_dir is None:
        pdf_dir = os.path.dirname(pdf_path)
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_base_dir = os.path.join(pdf_dir, f"{pdf_name}_output")

    output_base_dir = os.path.abspath(output_base_dir)
    logger.info("Output folder: %s", output_base_dir)
    os.makedirs(output_base_dir, exist_ok=True)

    processed_dir = os.path.join(output_base_dir, "processed")
    experiment_dir = os.path.join(output_base_dir, "experiment_results")
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(experiment_dir, exist_ok=True)

    logger.info("Streaming pages: convert -> normalize -> detect cells -> OCR")
    processed_count = 0
    converted_count = 0

    for page_num, image_path in iter_pdf_images(pdf_path, output_base_dir, dpi):
        converted_count += 1
        page_name = os.path.splitext(os.path.basename(image_path))[0]
        page_folder = f"{page_name}_steps"
        page_exp_dir = os.path.join(experiment_dir, page_folder)

        logger.info("Page %s: Processing %s", page_num, os.path.basename(image_path))
        _, steps = process_image_normalize_only(image_path, processed_dir)
        if not steps:
            logger.warning("Page %s: image normalization failed", page_num)
            continue

        normalized_path = os.path.join(_page_steps_dir(processed_dir, image_path), "02_normalized.png")
        if not os.path.exists(normalized_path):
            logger.warning("Page %s: normalized image was not created", page_num)
            continue

        if not process_normalized_image(normalized_path, page_exp_dir):
            logger.warning("Page %s: cell detection failed", page_num)
            continue

        result = process_cells_with_positions(page_exp_dir, experiment_dir)
        if result:
            processed_count += 1
            logger.info("Page %s: OCR completed", page_num)

    if converted_count == 0:
        logger.error("Failed to convert PDF")
        return None, None

    logger.info("Text extraction completed. Processed %s/%s pages", processed_count, converted_count)
    logger.info("Merging page results into JSON")

    json_pattern = os.path.join(experiment_dir, "*", "easyocr_results", "all_results.json")
    json_pattern = json_pattern.replace("\\", "/")
    json_files = glob.glob(json_pattern, recursive=True)
    logger.info("Found %s JSON files", len(json_files))

    if not json_files:
        logger.error("No JSON files found anywhere")
        return None, None

    merged_data = read_and_merge_json_files_simple(json_pattern)
    if not merged_data:
        logger.error("Failed to get merged data")
        return None, None

    output_json = _write_final_json(output_base_dir, merged_data)

    logger.info("Processing completed successfully")
    logger.info("Final JSON file: %s", output_json)
    logger.info("Number of pages: %s", len(merged_data))

    if merged_data:
        logger.debug("Sample of extracted data:")
        for page_name, page_data in list(merged_data.items())[:2]:
            logger.debug("Page %s: %s rows", page_name, len(page_data))
            if page_data:
                logger.debug("First row: %s", page_data[0])

    return output_json, merged_data
