"""Orchestrates the restoration + colorization pipeline on a single image.

Kept separate from app.py so the single-image and batch-processing UI
flows share exactly one implementation instead of two copies that could
drift apart, and so the orchestration is testable without Streamlit.
"""

import time

import cv2

from .colorizer import colorize
from .enhancer import enhance_image


def run_pipeline(image_rgb, net, denoise=True, contrast=True, sharpen=True, colorize_enabled=True):
    """Run the restoration + colorization pipeline on one RGB uint8 image.

    `net` is only used when colorize_enabled is True -- pass whatever
    load_model() returned (or a stub in tests). Ignored otherwise.

    Returns (result_rgb, elapsed_seconds).
    """
    start_time = time.perf_counter()

    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    enhanced = enhance_image(bgr, denoise=denoise, contrast=contrast, sharpen=sharpen)

    if colorize_enabled:
        result = colorize(enhanced, net)
        result = (result * 255).astype("uint8")
    else:
        result = enhanced

    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    elapsed_seconds = time.perf_counter() - start_time
    return result, elapsed_seconds
