"""Top-level pipeline orchestration."""

from __future__ import annotations

import glob
import json
import os

from ocr_pipeline.cell_detection import process_normalized_folder
from ocr_pipeline.merge import read_and_merge_json_files_simple
from ocr_pipeline.ocr_extraction import process_cells_with_positions
from ocr_pipeline.preprocessing import convert_pdf_to_images, process_folder_images


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

    print("\nStep 1: Converting PDF to images")
    image_paths, images_dir = convert_pdf_to_images(pdf_path, output_base_dir, dpi)
    if not image_paths:
        print("Failed to convert PDF")
        return None, None
    print(f"Successfully converted {len(image_paths)} pages")

    print("\nStep 2: Processing images")
    processed_dir = os.path.join(output_base_dir, "processed")
    process_folder_images(images_dir, processed_dir)
    print("Image processing completed")

    print("\nStep 3: Detecting cells")
    experiment_dir = os.path.join(output_base_dir, "experiment_results")
    os.makedirs(experiment_dir, exist_ok=True)
    process_normalized_folder(processed_dir, experiment_dir)
    print("Cell detection completed")

    print("\nStep 4: Extracting text with EasyOCR")
    processed_count = 0
    for folder in os.listdir(experiment_dir):
        folder_path = os.path.join(experiment_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        print(f"\nProcessing folder: {folder}")
        result = process_cells_with_positions(folder_path, experiment_dir)
        if result:
            processed_count += 1
            print(f"Successfully processed {folder}")

    print(f"\nText extraction completed. Processed {processed_count} folders")

    print("\nStep 5: Merging results into JSON")
    json_pattern = os.path.join(experiment_dir, "*", "easyocr_results", "all_results.json")
    json_pattern = json_pattern.replace("\\", "/")
    print(f"Searching for JSON files with pattern: {json_pattern}")

    json_files = glob.glob(json_pattern, recursive=True)
    print(f"Found {len(json_files)} JSON files")
    for json_file in json_files:
        print(f"  - {json_file}")

    if not json_files:
        print("\nTrying broader search...")
        all_json = glob.glob(os.path.join(experiment_dir, "**", "*.json"), recursive=True)
        if all_json:
            print("Found JSON files in different locations:")
            for path in all_json:
                print(f"  - {path}")
        else:
            print("No JSON files found anywhere")
            return None, None

    merged_data = read_and_merge_json_files_simple(json_pattern)
    if not merged_data:
        print("\nFailed to get merged data")
        return None, None

    output_json = os.path.join(output_base_dir, "final_extracted_text.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

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
