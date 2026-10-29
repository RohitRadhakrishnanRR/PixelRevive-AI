"""Tests for src/metrics.py."""

import cv2
import numpy as np
import pytest

from src.metrics import (
    estimate_brightness,
    estimate_contrast,
    estimate_sharpness,
    get_processing_device,
)


class TestEstimateSharpness:
    def test_sharp_edge_scores_higher_than_blurred(self):
        sharp = np.zeros((200, 200, 3), dtype=np.uint8)
        sharp[:, 100:] = 255
        blurred = cv2.GaussianBlur(sharp, (25, 25), 0)

        assert estimate_sharpness(sharp) > estimate_sharpness(blurred) * 5

    def test_flat_image_scores_near_zero(self):
        flat = np.full((100, 100, 3), 128, dtype=np.uint8)
        assert estimate_sharpness(flat) < 1.0

    def test_accepts_grayscale_input(self):
        gray = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        # should not raise on a 2D array
        estimate_sharpness(gray)


class TestEstimateContrast:
    def test_high_contrast_scores_higher_than_low(self):
        high = np.zeros((100, 100, 3), dtype=np.uint8)
        high[:, 50:] = 255
        low = np.full((100, 100, 3), 120, dtype=np.uint8)
        low[:, 50:] = 135

        assert estimate_contrast(high) > estimate_contrast(low) * 5

    def test_flat_image_has_zero_contrast(self):
        flat = np.full((50, 50, 3), 128, dtype=np.uint8)
        assert estimate_contrast(flat) == 0.0


class TestEstimateBrightness:
    @pytest.mark.parametrize("value", [0, 64, 128, 200, 255])
    def test_matches_known_flat_values(self, value):
        image = np.full((30, 30, 3), value, dtype=np.uint8)
        assert estimate_brightness(image) == pytest.approx(value, abs=0.5)

    def test_orders_correctly_across_brightness_levels(self):
        dark = np.full((30, 30, 3), 20, dtype=np.uint8)
        mid = np.full((30, 30, 3), 128, dtype=np.uint8)
        bright = np.full((30, 30, 3), 235, dtype=np.uint8)
        assert estimate_brightness(dark) < estimate_brightness(mid) < estimate_brightness(bright)


class TestGetProcessingDevice:
    def test_returns_cpu_or_cuda_and_never_raises(self):
        device = get_processing_device()
        assert device in ("CPU", "CUDA")

    def test_reports_cpu_on_this_opencv_headless_build(self):
        # opencv-python-headless ships without CUDA support, so this
        # should never claim CUDA is available in this project's environment
        assert get_processing_device() == "CPU"
