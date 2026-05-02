"""Pipeline constants and tunables."""

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

EXPERIMENTAL_CONFIGS = [
    {
        "name": "The first experience",
        "min_cell_width": 20,
        "min_cell_height": 15,
        "max_cell_area_ratio": 0.25,
        "line_kernel_size": 30,
        "ocr_psm": 7,
        "ocr_oem": 3,
        "cell_margin": 5,
        "clahe_clip_limit": 3.0,
        "clahe_grid_size": 8,
        "denoising_strength": 12,
        "row_threshold": 25,
        "col_threshold": 50,
    }
]
