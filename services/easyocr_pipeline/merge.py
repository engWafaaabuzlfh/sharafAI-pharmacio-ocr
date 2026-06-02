"""Merge per-page JSON artifacts into one result payload."""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path


def read_and_merge_json_files_simple(base_pattern: str):
    print(f"Searching for JSON files with pattern: {base_pattern}")
    json_files = glob.glob(base_pattern, recursive=True)

    print(f"Found {len(json_files)} JSON files")
    for file_path in json_files:
        print(f"  - {file_path}")

    if not json_files:
        print("No JSON files found")
        return None

    merged_data = {}
    for json_file in sorted(json_files):
        page_name = Path(json_file).parent.parent.name
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            print(f"Processing page: {page_name}")
            page_rows = {}
            for cell_file, cell_info in data.items():
                cell_match = re.search(r"cell_(\d+)", cell_file)
                if not cell_match:
                    continue

                cell_num = int(cell_match.group(1))
                position = cell_info.get("position", {})
                page_rows[cell_num] = {
                    "text": cell_info.get("text", ""),
                    "x": position.get("x", 0),
                    "y": position.get("y", 0),
                    "cell_num": cell_num,
                }

            if not page_rows:
                print(f"Warning: No cells found in {page_name}")
                continue

            sorted_cells = sorted(page_rows.values(), key=lambda c: (c["y"], c["x"]))
            rows = []
            current_row = []
            current_y = None
            row_threshold = 50

            for cell in sorted_cells:
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

            page_data = []
            for row_cells in rows:
                row_dict = {}
                for col_idx, cell in enumerate(row_cells, 1):
                    row_dict[f"Col_{col_idx}"] = cell["text"]
                page_data.append(row_dict)

            merged_data[page_name] = page_data
            print(f"Added {len(page_data)} rows from {page_name}")
        except Exception as e:
            print(f"Error in {json_file}: {e}")
            merged_data[page_name] = []

    print(f"Total pages merged: {len(merged_data)}")
    return merged_data
