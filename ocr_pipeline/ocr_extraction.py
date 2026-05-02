"""OCR extraction and row/column structuring."""

from __future__ import annotations

import json
import os

import cv2
import numpy as np
import pandas as pd

from ocr_pipeline.runtime import get_reader


def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return tuple(convert_to_serializable(i) for i in obj)
    if isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    return obj


def fix_slash_order(text: str):
    if "/" not in text and "\\" not in text:
        return text
    slash = "/" if "/" in text else "\\"
    parts = text.split(slash)
    if len(parts) != 2:
        return text

    left_words = parts[0].strip().split()
    right_words = parts[1].strip().split()
    if len(right_words) <= 2:
        return text

    new_left = " ".join(right_words[-2:])
    new_right = " ".join(left_words + right_words[:-2])
    return f"{new_right} {slash} {new_left}"


def extract_arabic_from_cell_image(image_path: str, debug_dir: str | None = None):
    image = cv2.imread(image_path)
    results = get_reader().readtext(image_path, paragraph=True, detail=1)

    words = []
    details = []
    for res in results:
        bbox = res[0]
        text = res[1]
        conf = res[2] if len(res) > 2 else 1.0

        x_center = sum([p[0] for p in bbox]) / 4
        y_center = sum([p[1] for p in bbox]) / 4
        words.append({"text": text, "x": x_center, "y": y_center, "bbox": bbox})
        details.append({"bbox": bbox, "text": text, "confidence": float(conf)})

    full_text = fix_slash_order(" ".join([w["text"] for w in words]))

    if debug_dir and image is not None:
        os.makedirs(debug_dir, exist_ok=True)
        debug_img = image.copy()
        row_threshold = 10
        rows_words = []
        current_row = []
        current_y = None

        for word in sorted(words, key=lambda w: w["y"]):
            if current_y is None or abs(word["y"] - current_y) <= row_threshold:
                current_row.append(word)
                if current_y is None:
                    current_y = word["y"]
            else:
                rows_words.append(current_row)
                current_row = [word]
                current_y = word["y"]

        if current_row:
            rows_words.append(current_row)

        for idx, row_words in enumerate(rows_words, 1):
            all_x = [p[0] for w in row_words for p in w["bbox"]]
            all_y = [p[1] for w in row_words for p in w["bbox"]]
            x_min, x_max = int(min(all_x)), int(max(all_x))
            y_min, y_max = int(min(all_y)), int(max(all_y))
            cv2.rectangle(debug_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(
                debug_img,
                str(idx),
                (x_min, y_min - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        debug_path = os.path.join(debug_dir, os.path.basename(image_path))
        cv2.imwrite(debug_path, debug_img)

    return full_text, details


def read_cell_positions(experiment_dir: str):
    csv_path = os.path.join(experiment_dir, "cell_positions.csv")
    if not os.path.exists(csv_path):
        print(f"Cell positions file not found: {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    df["cell_num"] = df["image_file"].str.extract(r"cell_(\d+)").astype(int)
    return df.sort_values(["y", "x"])


def group_cells_by_row(df: pd.DataFrame, row_threshold: int = 50):
    rows = []
    current_row = []
    current_y = None
    for _, cell in df.iterrows():
        if current_y is None or abs(cell["y"] - current_y) <= row_threshold:
            current_row.append(cell)
            if current_y is None:
                current_y = cell["y"]
        else:
            rows.append(sorted(current_row, key=lambda c: c["x"]))
            current_row = [cell]
            current_y = cell["y"]
    if current_row:
        rows.append(sorted(current_row, key=lambda c: c["x"]))
    return rows


def process_cells_with_positions(experiment_dir: str, res_root: str):
    cells_dir = os.path.join(experiment_dir, "cells")
    folder_name = os.path.basename(experiment_dir)
    output_dir = os.path.join(res_root, folder_name, "easyocr_results")
    debug_dir = os.path.join(output_dir, "debug_boxes")

    print(f"Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    df_positions = read_cell_positions(experiment_dir)
    if df_positions is None:
        return None

    cell_texts = {}
    all_results = {}

    for _, row in df_positions.iterrows():
        cell_file = os.path.basename(row["image_file"])
        cell_path = os.path.join(cells_dir, cell_file)
        if not os.path.exists(cell_path):
            continue

        text, details = extract_arabic_from_cell_image(cell_path, debug_dir=debug_dir)
        txt_file = os.path.join(output_dir, cell_file.replace(".jpg", ".txt"))
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(text)

        cell_texts[row["cell_num"]] = text
        all_results[cell_file] = {
            "text": text,
            "position": {
                "x": int(row["x"]),
                "y": int(row["y"]),
                "width": int(row["width"]),
                "height": int(row["height"]),
            },
            "details": convert_to_serializable(details),
        }

    rows = group_cells_by_row(df_positions)
    rows_data = []
    for row_idx, row_cells in enumerate(rows, 1):
        row_dict = {"Row": row_idx}
        for col_idx, cell in enumerate(row_cells, 1):
            row_dict[f"Col_{col_idx}"] = cell_texts.get(cell["cell_num"], "")
        rows_data.append(row_dict)

    df_rows = pd.DataFrame(rows_data)
    df_rows.to_csv(
        os.path.join(output_dir, "extracted_texts_by_row.csv"), index=False, encoding="utf-8-sig"
    )
    df_rows.to_excel(
        os.path.join(output_dir, "extracted_texts_by_row.xlsx"), index=False, engine="openpyxl"
    )

    simple_data = []
    for _, row in df_positions.iterrows():
        simple_data.append(
            {
                "cell_id": row["cell_id"],
                "cell_num": row["cell_num"],
                "x": row["x"],
                "y": row["y"],
                "extracted_text": cell_texts.get(row["cell_num"], ""),
            }
        )
    pd.DataFrame(simple_data).to_csv(
        os.path.join(output_dir, "extracted_texts_simple.csv"), index=False, encoding="utf-8-sig"
    )

    json_path = os.path.join(output_dir, "all_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"JSON saved: {json_path}")
    print(f"Processed: {experiment_dir}")
    print(f"Results saved in: {output_dir}\n")
    return all_results, df_rows
