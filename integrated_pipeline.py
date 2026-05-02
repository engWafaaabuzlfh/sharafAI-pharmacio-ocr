"""Backward-compatible exports for the modular OCR pipeline."""

from ocr_pipeline.cell_detection import (
    detect_cells,
    preprocess_image,
    process_normalized_folder,
    save_cell_positions,
)
from ocr_pipeline.merge import read_and_merge_json_files_simple
from ocr_pipeline.ocr_extraction import (
    convert_to_serializable,
    extract_arabic_from_cell_image,
    fix_slash_order,
    group_cells_by_row,
    process_cells_with_positions,
    read_cell_positions,
)
from ocr_pipeline.pipeline import process_pdf_to_json
from ocr_pipeline.preprocessing import (
    convert_pdf_to_images,
    load_image_rgb,
    normalize_colors,
    process_folder_images,
    process_image_normalize_only,
    save_results,
)
from ocr_pipeline.settings import EXPERIMENTAL_CONFIGS, IMAGE_EXTENSIONS

__all__ = [
    "IMAGE_EXTENSIONS",
    "EXPERIMENTAL_CONFIGS",
    "convert_pdf_to_images",
    "load_image_rgb",
    "normalize_colors",
    "process_image_normalize_only",
    "save_results",
    "process_folder_images",
    "preprocess_image",
    "detect_cells",
    "save_cell_positions",
    "process_normalized_folder",
    "convert_to_serializable",
    "fix_slash_order",
    "extract_arabic_from_cell_image",
    "read_cell_positions",
    "group_cells_by_row",
    "process_cells_with_positions",
    "read_and_merge_json_files_simple",
    "process_pdf_to_json",
]
