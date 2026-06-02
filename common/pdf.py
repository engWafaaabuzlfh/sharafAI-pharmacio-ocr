"""Shared PDF rendering helpers."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def render_pdf_pages(pdf_path: Path, *, dpi: int = 180) -> list[Image.Image]:
    """Render PDF pages with PyMuPDF, avoiding the external Poppler dependency."""
    import fitz

    scale = dpi / 72
    matrix = fitz.Matrix(scale, scale)
    pages: list[Image.Image] = []

    with fitz.open(str(pdf_path)) as doc:
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pages.append(image)
    return pages
