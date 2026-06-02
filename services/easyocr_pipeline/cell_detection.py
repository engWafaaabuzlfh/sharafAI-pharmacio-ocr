"""Table cell detection and cell image extraction."""

from __future__ import annotations

import csv
import os

import cv2
import numpy as np

from services.easyocr_pipeline.settings import EXPERIMENTAL_CONFIGS


def preprocess_image(img, config: dict):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    clahe = cv2.createCLAHE(
        clipLimit=config["clahe_clip_limit"],
        tileGridSize=(config["clahe_grid_size"], config["clahe_grid_size"]),
    )
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, h=config["denoising_strength"])
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5
    )
    return binary, gray, enhanced


def detect_cells(binary, original_shape: tuple[int, int], config: dict):
    height, width = original_shape
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (config["line_kernel_size"], 1)
    )
    horizontal_lines = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, config["line_kernel_size"])
    )
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    table_structure = cv2.add(horizontal_lines, vertical_lines)

    kernel = np.ones((3, 3), np.uint8)
    table_structure = cv2.dilate(table_structure, kernel, iterations=2)
    table_structure = cv2.erode(table_structure, kernel, iterations=1)

    contours, _ = cv2.findContours(table_structure, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cells = []
    max_area = width * height * config["max_cell_area_ratio"]
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area > 100 and area < max_area:
            if w >= config["min_cell_width"] and h >= config["min_cell_height"]:
                cells.append((x, y, w, h))
    return cells


def save_cell_positions(cells: list[tuple[int, int, int, int]], exp_dir: str):
    csv_path = os.path.join(exp_dir, "cell_positions.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cell_id", "x", "y", "width", "height", "image_file"])
        for i, (x, y, w, h) in enumerate(cells):
            writer.writerow([i + 1, x, y, w, h, f"cells/cell_{i:04d}.jpg"])
    return csv_path


def _fallback_grid_detection(binary, gray_shape: tuple[int, int], config: dict):
    h_projection = np.sum(binary, axis=1) / 255
    v_projection = np.sum(binary, axis=0) / 255

    rows = []
    in_row = False
    for i, val in enumerate(h_projection):
        if val > gray_shape[1] * 0.05 and not in_row:
            in_row = True
            start = i
        elif val <= gray_shape[1] * 0.05 and in_row:
            in_row = False
            if i - start > config["min_cell_height"]:
                rows.append((start, i))

    cols = []
    in_col = False
    for i, val in enumerate(v_projection):
        if val > gray_shape[0] * 0.05 and not in_col:
            in_col = True
            start = i
        elif val <= gray_shape[0] * 0.05 and in_col:
            in_col = False
            if i - start > config["min_cell_width"]:
                cols.append((start, i))

    if not rows or not cols:
        return []

    cells = []
    for y1, y2 in rows:
        for x1, x2 in cols:
            cells.append((x1, y1, x2 - x1, y2 - y1))
    return cells


def process_normalized_folder(input_dir: str, output_dir: str):
    if not os.path.exists(input_dir):
        print(f"Input folder not found: {input_dir}")
        return

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file != "02_normalized.png":
                continue

            img_path = os.path.join(root, file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Cannot read image: {img_path}")
                continue

            config = EXPERIMENTAL_CONFIGS[0]
            relative_path = os.path.relpath(root, input_dir)
            exp_dir = os.path.join(output_dir, relative_path)
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
