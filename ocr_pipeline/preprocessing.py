"""Image loading and normalization helpers."""

from __future__ import annotations

import os

import cv2
import numpy as np

from ocr_pipeline.settings import IMAGE_EXTENSIONS


def convert_pdf_to_images(pdf_path: str, output_dir: str, dpi: int = 300):
    from pdf2image import convert_from_path

    raw_images_dir = os.path.join(output_dir, "seperate_image")
    os.makedirs(raw_images_dir, exist_ok=True)

    try:
        images = convert_from_path(pdf_path, dpi=dpi)
        print(f"Number of extracted images: {len(images)}")

        image_paths = []
        for page_num, image in enumerate(images, start=1):
            filename = f"page_{page_num:03d}_raw.jpg"
            filepath = os.path.join(raw_images_dir, filename)
            image.save(filepath, "JPEG", quality=95, optimize=True)
            image_paths.append(filepath)

        return image_paths, raw_images_dir
    except Exception as e:
        print(f"Error converting PDF: {e}")
        return None, None


def load_image_rgb(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise Exception(f"Cannot load image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"Image dimensions: {img_rgb.shape}")
    return img_rgb


def normalize_colors(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    p2, p98 = np.percentile(gray, (2, 98))
    stretched = np.clip((gray - p2) * (255.0 / (p98 - p2)), 0, 255).astype(np.uint8)
    return stretched


def save_results(steps: dict, image_path: str, output_dir: str):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    result_dir = os.path.join(output_dir, f"{image_name}_steps")
    os.makedirs(result_dir, exist_ok=True)

    print(f"\nSaving results in: {result_dir}")
    for step_name, step_image in steps.items():
        if step_image is None:
            continue
        output_path = os.path.join(result_dir, f"{step_name}.png")
        if len(step_image.shape) == 2:
            cv2.imwrite(output_path, step_image)
        else:
            cv2.imwrite(output_path, cv2.cvtColor(step_image, cv2.COLOR_RGB2BGR))


def process_image_normalize_only(image_path: str, output_dir: str | None = None):
    steps = {}
    try:
        img_rgb = load_image_rgb(image_path)
        steps["01_original"] = img_rgb

        normalized = normalize_colors(img_rgb)
        steps["02_normalized"] = normalized

        if output_dir:
            save_results(steps, image_path, output_dir)

        return normalized, steps
    except Exception as e:
        print(f"\nError: {e}")
        return None, None


def process_folder_images(input_dir: str, output_dir: str):
    if not os.path.exists(input_dir):
        print(f"Input folder not found: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    image_files = [
        f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ]

    if not image_files:
        print("No images found in folder.")
        return

    print(f"Found {len(image_files)} images. Processing...")
    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        print(f"\nProcessing: {img_file}")
        process_image_normalize_only(img_path, output_dir)

    print("\nAll images processed.")
