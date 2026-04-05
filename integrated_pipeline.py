import os
import numpy as np
import pandas as pd
from PIL import Image
from pdf2image import convert_from_path
import glob
import cv2
import csv
import json
from pathlib import Path
import re
import easyocr
import warnings
warnings.filterwarnings('ignore')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
reader = easyocr.Reader(['ar', 'en'], gpu=False)

IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']

EXPERIMENTAL_CONFIGS = [
    {
        'name': "The first experience",
        'min_cell_width': 20,
        'min_cell_height': 15,
        'max_cell_area_ratio': 0.25,
        'line_kernel_size': 30,
        'ocr_psm': 7,
        'ocr_oem': 3,
        'cell_margin': 5,
        'clahe_clip_limit': 3.0,
        'clahe_grid_size': 8,
        'denoising_strength': 12,
        'row_threshold': 25,
        'col_threshold': 50,
    }
]

def convert_pdf_to_images(pdf_path, output_dir, dpi=300):
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    raw_images_dir = os.path.join(output_dir, "seperate_image")
    os.makedirs(raw_images_dir, exist_ok=True)

    try:
        images = convert_from_path(pdf_path, dpi=dpi)
        print(f"Number of extracted images: {len(images)}")

        image_paths = []
        for page_num, image in enumerate(images, start=1):
            filename = f"page_{page_num:03d}_raw.jpg"
            filepath = os.path.join(raw_images_dir, filename)
            image.save(filepath, 'JPEG', quality=95, optimize=True)
            image_paths.append(filepath)

        return image_paths, raw_images_dir

    except Exception as e:
        print(f"Error converting PDF: {e}")
        return None, None


def load_image_rgb(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise Exception(f"Cannot load image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"Image dimensions: {img_rgb.shape}")
    return img_rgb


def normalize_colors(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    p2, p98 = np.percentile(gray, (2, 98))
    stretched = np.clip((gray - p2) * (255.0 / (p98 - p2)),
                        0, 255).astype(np.uint8)
    return stretched


def process_image_normalize_only(image_path, output_dir=None):
    steps = {}
    try:
        img_rgb = load_image_rgb(image_path)
        steps['01_original'] = img_rgb

        normalized = normalize_colors(img_rgb)
        steps['02_normalized'] = normalized

        if output_dir:
            save_results(steps, image_path, output_dir)

        return normalized, steps

    except Exception as e:
        print(f"\nError: {e}")
        return None, None


def save_results(steps, image_path, output_dir):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    result_dir = os.path.join(output_dir, f"{image_name}_steps")
    os.makedirs(result_dir, exist_ok=True)

    print(f"\nSaving results in: {result_dir}")

    for step_name, step_image in steps.items():
        if step_image is not None:
            if len(step_image.shape) == 2:
                output_path = os.path.join(result_dir, f"{step_name}.png")
                cv2.imwrite(output_path, step_image)
            else:
                output_path = os.path.join(result_dir, f"{step_name}.png")
                cv2.imwrite(output_path, cv2.cvtColor(
                    step_image, cv2.COLOR_RGB2BGR))


def process_folder_images(input_dir, output_dir):
    if not os.path.exists(input_dir):
        print(f"Input folder not found: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[
        1].lower() in IMAGE_EXTENSIONS]

    if not image_files:
        print("No images found in folder.")
        return

    print(f"Found {len(image_files)} images. Processing...")

    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        print(f"\nProcessing: {img_file}")
        normalized, steps = process_image_normalize_only(img_path, output_dir)

    print("\nAll images processed.")


def preprocess_image(img, config):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    clahe = cv2.createCLAHE(clipLimit=config['clahe_clip_limit'],
                            tileGridSize=(config['clahe_grid_size'], config['clahe_grid_size']))
    enhanced = clahe.apply(gray)

    denoised = cv2.fastNlMeansDenoising(
        enhanced, h=config['denoising_strength'])

    binary = cv2.adaptiveThreshold(denoised, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 5)

    return binary, gray, enhanced


def detect_cells(binary, original_shape, config):
    height, width = original_shape

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (config['line_kernel_size'], 1))
    horizontal_lines = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, config['line_kernel_size']))
    vertical_lines = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    table_structure = cv2.add(horizontal_lines, vertical_lines)

    kernel = np.ones((3, 3), np.uint8)
    table_structure = cv2.dilate(table_structure, kernel, iterations=2)
    table_structure = cv2.erode(table_structure, kernel, iterations=1)

    contours, _ = cv2.findContours(
        table_structure, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cells = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        max_area = width * height * config['max_cell_area_ratio']

        if area > 100 and area < max_area:
            if w >= config['min_cell_width'] and h >= config['min_cell_height']:
                cells.append((x, y, w, h))

    return cells


def save_cell_positions(cells, cells_dir, exp_dir):
    csv_path = os.path.join(exp_dir, "cell_positions.csv")

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['cell_id', 'x', 'y', 'width', 'height', 'image_file'])

        for i, (x, y, w, h) in enumerate(cells):
            writer.writerow([i+1, x, y, w, h, f"cells/cell_{i:04d}.jpg"])

    return csv_path


def process_normalized_folder(input_dir, output_dir):
    if not os.path.exists(input_dir):
        print(f"Input folder not found: {input_dir}")
        return

    for root, dirs, files in os.walk(input_dir):
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
                h_projection = np.sum(binary, axis=1) / 255
                v_projection = np.sum(binary, axis=0) / 255

                rows = []
                in_row = False
                for i, val in enumerate(h_projection):
                    if val > gray.shape[1] * 0.05 and not in_row:
                        in_row = True
                        start = i
                    elif val <= gray.shape[1] * 0.05 and in_row:
                        in_row = False
                        if i - start > config['min_cell_height']:
                            rows.append((start, i))

                cols = []
                in_col = False
                for i, val in enumerate(v_projection):
                    if val > gray.shape[0] * 0.05 and not in_col:
                        in_col = True
                        start = i
                    elif val <= gray.shape[0] * 0.05 and in_col:
                        in_col = False
                        if i - start > config['min_cell_width']:
                            cols.append((start, i))

                if len(rows) > 0 and len(cols) > 0:
                    cells = []
                    for y1, y2 in rows:
                        for x1, x2 in cols:
                            cells.append((x1, y1, x2-x1, y2-y1))

            debug_img = img.copy()
            for x, y, w, h in cells:
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 1)
            cv2.imwrite(os.path.join(
                debug_dir, "detected_cells.jpg"), debug_img)

            cells = sorted(cells, key=lambda b: (b[1], b[0]))
            for i, (x, y, w, h) in enumerate(cells):
                margin = config['cell_margin']
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(gray.shape[1], x + w + margin)
                y2 = min(gray.shape[0], y + h + margin)

                cell_img = enhanced[y1:y2, x1:x2]
                cv2.imwrite(os.path.join(
                    cells_dir, f"cell_{i:04d}.jpg"), cell_img)

            save_cell_positions(cells, cells_dir, exp_dir)
            print(f"Processed and saved: {img_path}")


def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(i) for i in obj)
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    return obj


def fix_slash_order(text):
    if "/" not in text and "\\" not in text:
        return text
    slash = "/" if "/" in text else "\\"
    parts = text.split(slash)
    if len(parts) != 2:
        return text
    left = parts[0].strip()
    right = parts[1].strip()
    left_words = left.split()
    right_words = right.split()
    if len(right_words) > 2:
        new_left = " ".join(right_words[-2:])
        new_right = " ".join(left_words + right_words[:-2])
        return f"{new_right} {slash} {new_left}"
    return text


def extract_arabic_from_cell_image(image_path, debug_dir=None):
    image = cv2.imread(image_path)
    results = reader.readtext(image_path, paragraph=True, detail=1)

    words = []
    details = []

    for res in results:
        bbox = res[0]
        text = res[1]
        conf = res[2] if len(res) > 2 else 1.0

        x_center = sum([p[0] for p in bbox]) / 4
        y_center = sum([p[1] for p in bbox]) / 4

        words.append({"text": text, "x": x_center,
                     "y": y_center, "bbox": bbox})
        details.append({"bbox": bbox, "text": text, "confidence": float(conf)})

    full_text = " ".join([w["text"] for w in words])
    full_text = fix_slash_order(full_text)

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        debug_img = image.copy()
        row_threshold = 10
        rows_words = []
        current_row = []
        current_y = None
        for w in sorted(words, key=lambda w: w["y"]):
            if current_y is None or abs(w["y"] - current_y) <= row_threshold:
                current_row.append(w)
                if current_y is None:
                    current_y = w["y"]
            else:
                rows_words.append(current_row)
                current_row = [w]
                current_y = w["y"]
        if current_row:
            rows_words.append(current_row)

        for idx, row_words in enumerate(rows_words, 1):
            all_x = [p[0] for w in row_words for p in w["bbox"]]
            all_y = [p[1] for w in row_words for p in w["bbox"]]
            x_min, x_max = int(min(all_x)), int(max(all_x))
            y_min, y_max = int(min(all_y)), int(max(all_y))
            cv2.rectangle(debug_img, (x_min, y_min),
                          (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(debug_img, str(idx), (x_min, y_min-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        debug_path = os.path.join(debug_dir, os.path.basename(image_path))
        cv2.imwrite(debug_path, debug_img)

    return full_text, details


def read_cell_positions(experiment_dir):
    csv_path = os.path.join(experiment_dir, "cell_positions.csv")
    if not os.path.exists(csv_path):
        print(f"Cell positions file not found: {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    df["cell_num"] = df["image_file"].str.extract(r"cell_(\d+)").astype(int)
    df_sorted = df.sort_values(["y", "x"])
    return df_sorted


def group_cells_by_row(df, row_threshold=50):
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


def process_cells_with_positions(experiment_dir, res_root):
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
        if os.path.exists(cell_path):
            text, details = extract_arabic_from_cell_image(
                cell_path, debug_dir=debug_dir)
            txt_file = os.path.join(
                output_dir, cell_file.replace(".jpg", ".txt"))
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write(text)

            cell_texts[row["cell_num"]] = text
            all_results[cell_file] = {
                "text": text,
                "position": {
                    "x": int(row["x"]),
                    "y": int(row["y"]),
                    "width": int(row["width"]),
                    "height": int(row["height"])
                },
                "details": convert_to_serializable(details)
            }

    rows = group_cells_by_row(df_positions)
    rows_data = []
    for row_idx, row_cells in enumerate(rows, 1):
        row_dict = {"Row": row_idx}
        for col_idx, cell in enumerate(row_cells, 1):
            cell_text = cell_texts.get(cell["cell_num"], "")
            row_dict[f"Col_{col_idx}"] = cell_text
        rows_data.append(row_dict)

    df_rows = pd.DataFrame(rows_data)
    df_rows.to_csv(os.path.join(
        output_dir, "extracted_texts_by_row.csv"), index=False, encoding="utf-8-sig")
    df_rows.to_excel(os.path.join(
        output_dir, "extracted_texts_by_row.xlsx"), index=False, engine="openpyxl")

    simple_data = []
    for _, row in df_positions.iterrows():
        simple_data.append({
            "cell_id": row["cell_id"],
            "cell_num": row["cell_num"],
            "x": row["x"],
            "y": row["y"],
            "extracted_text": cell_texts.get(row["cell_num"], "")
        })
    pd.DataFrame(simple_data).to_csv(os.path.join(output_dir, "extracted_texts_simple.csv"),
                                     index=False, encoding="utf-8-sig")

    # حفظ ملف JSON
    json_path = os.path.join(output_dir, "all_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"JSON saved: {json_path}")
    print(f"Processed: {experiment_dir}")
    print(f"Results saved in: {output_dir}\n")

    return all_results, df_rows

def read_and_merge_json_files_simple(base_pattern):
    print(f"Searching for JSON files with pattern: {base_pattern}")
    json_files = glob.glob(base_pattern, recursive=True)

    print(f"Found {len(json_files)} JSON files")
    for f in json_files:
        print(f"  - {f}")

    if not json_files:
        print("No JSON files found")
        return None

    merged_data = {}

    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            page_name = Path(json_file).parent.parent.name
            print(f"Processing page: {page_name}")

            page_rows = {}

            for cell_file, cell_info in data.items():
                cell_match = re.search(r'cell_(\d+)', cell_file)
                if cell_match:
                    cell_num = int(cell_match.group(1))

                    text = cell_info.get('text', '')

                    position = cell_info.get('position', {})
                    x = position.get('x', 0)
                    y = position.get('y', 0)

                    if cell_num not in page_rows:
                        page_rows[cell_num] = {
                            'text': text,
                            'x': x,
                            'y': y,
                            'cell_num': cell_num
                        }

            if not page_rows:
                print(f"Warning: No cells found in {page_name}")
                continue

            sorted_cells = sorted(page_rows.values(),
                                  key=lambda c: (c['y'], c['x']))

            rows = []
            current_row = []
            current_y = None
            row_threshold = 50

            for cell in sorted_cells:
                if current_y is None or abs(cell['y'] - current_y) <= row_threshold:
                    current_row.append(cell)
                    if current_y is None:
                        current_y = cell['y']
                else:
                    rows.append(sorted(current_row, key=lambda c: c['x']))
                    current_row = [cell]
                    current_y = cell['y']

            if current_row:
                rows.append(sorted(current_row, key=lambda c: c['x']))

            page_data = []

            for row_idx, row_cells in enumerate(rows, 1):
                row_dict = {}
                for col_idx, cell in enumerate(row_cells, 1):
                    row_dict[f'Col_{col_idx}'] = cell['text']
                page_data.append(row_dict)

            merged_data[page_name] = page_data
            print(f"Added {len(page_data)} rows from {page_name}")

        except Exception as e:
            print(f"Error in {json_file}: {e}")
            merged_data[page_name] = []

    print(f"Total pages merged: {len(merged_data)}")
    return merged_data


def process_pdf_to_json(pdf_path, output_base_dir=None, dpi=300):
    print("="*70)
    print("Starting PDF processing and text extraction")
    print("="*70)

    if output_base_dir is None:
        pdf_dir = os.path.dirname(pdf_path)
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_base_dir = os.path.join(pdf_dir, f"{pdf_name}_output")

    output_base_dir = os.path.abspath(output_base_dir)
    print(f"\nOutput folder: {output_base_dir}")

    os.makedirs(output_base_dir, exist_ok=True)

    print("\nStep 1: Converting PDF to images")
    image_paths, images_dir = convert_pdf_to_images(
        pdf_path, output_base_dir, dpi)

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
        if os.path.isdir(folder_path):
            print(f"\nProcessing folder: {folder}")
            result = process_cells_with_positions(folder_path, experiment_dir)
            if result:
                processed_count += 1
                print(f"Successfully processed {folder}")

    print(f"\nText extraction completed. Processed {processed_count} folders")

    print("\nStep 5: Merging results into JSON")

    json_pattern = os.path.join(
        experiment_dir, "*", "easyocr_results", "all_results.json")
    json_pattern = json_pattern.replace("\\", "/")
    print(f"Searching for JSON files with pattern: {json_pattern}")

    json_files = glob.glob(json_pattern, recursive=True)
    print(f"Found {len(json_files)} JSON files")

    for jf in json_files:
        print(f"  - {jf}")

    if not json_files:
        print("\nTrying broader search...")
        all_json = glob.glob(os.path.join(
            experiment_dir, "**", "*.json"), recursive=True)
        if all_json:
            print(f"Found JSON files in different locations:")
            for j in all_json:
                print(f"  - {j}")
        else:
            print("No JSON files found anywhere")
            return None, None

    merged_data = read_and_merge_json_files_simple(json_pattern)

    if merged_data:
        output_json = os.path.join(
            output_base_dir, "final_extracted_text.json")
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)

        print(f"Processing completed successfully!")
        print(f"Final JSON file: {output_json}")
        print(f"Number of pages: {len(merged_data)}")

        if merged_data:
            print("\nSample of extracted data:")
            for page_name, page_data in list(merged_data.items())[:2]:
                print(f"\n  {page_name}: {len(page_data)} rows")
                if page_data and len(page_data) > 0:
                    print(f"  First row: {page_data[0]}")


        return output_json, merged_data
    else:
        print("\nFailed to get merged data")
        return None, None


# if __name__ == "__main__":
#     result_json, extracted_data = process_pdf_to_json("test.pdf")
#     print(f"ther result in : {result_json}")
