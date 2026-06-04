"""JSON parsing and writing helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as first_error:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as second_error:
                return {
                    "raw_response": text,
                    "warnings": [
                        "Response was not valid JSON",
                        f"first_error={first_error.msg} at pos {first_error.pos}",
                        f"second_error={second_error.msg} at pos {second_error.pos}",
                    ],
                }

        return {"raw_response": text, "warnings": [f"Response was not valid JSON: {first_error.msg}"]}


def write_json(data: dict[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def dumps_json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)
