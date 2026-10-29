"""Tests for src/enhancer.py."""

import numpy as np
import pytest

from src.config import SHARPEN_KERNEL
from src.enhancer import denoise_image, enhance_contrast, enhance_image, sharpen_image


class TestEnhanceImageToggling:
    """Each restoration stage must be independently on/off -- this is the
    whole point of the sidebar toggles."""

    def test_all_stages_disabled_is_a_true_noop(self, sample_image):
        result = enhance_image(sample_image, denoise=False, contrast=False, sharpen=False)
        assert np.array_equal(result, sample_image)

    def test_default_call_enables_all_stages(self, sample_image):
        result = enhance_image(sample_image)
        assert not np.array_equal(result, sample_image)

    @pytest.mark.parametrize("flags", [
        {"denoise": True, "contrast": False, "sharpen": False},
        {"denoise": False, "contrast": True, "sharpen": False},
        {"denoise": False, "contrast": False, "sharpen": True},
    ])
    def test_each_stage_runs_independently_without_error(self, sample_image, flags):
        result = enhance_image(sample_image, **flags)
        assert result.shape == sample_image.shape
        assert result.dtype == sample_image.dtype

    def test_preserves_shape_and_dtype(self, sample_image):
        result = enhance_image(sample_image)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8


class TestSharpenBrightnessRegression:
    """Regression test for a P0 bug: the sharpen kernel summed to 1.5
    instead of 1.0, inflating brightness on flat regions by ~50%."""

    def test_kernel_sums_to_one(self):
        assert SHARPEN_KERNEL.sum() == 1

    def test_flat_region_brightness_is_preserved(self, flat_image):
        result = sharpen_image(flat_image)
        assert abs(int(result.mean()) - int(flat_image.mean())) <= 3


class TestContrastEnhancement:
    def test_clahe_increases_contrast_on_low_contrast_image(self):
        low_contrast = np.full((80, 80, 3), 120, dtype=np.uint8)
        low_contrast[:, 40:] = 135
        before_std = low_contrast.std()
        after_std = enhance_contrast(low_contrast).std()
        assert after_std > before_std

    def test_preserves_shape_and_dtype(self, sample_image):
        result = enhance_contrast(sample_image)
        assert result.shape == sample_image.shape
        assert result.dtype == sample_image.dtype


class TestDenoise:
    def test_reduces_variance_on_a_noisy_flat_region(self):
        rng = np.random.default_rng(1)
        noisy = np.full((80, 80, 3), 128, dtype=np.int16)
        noisy += rng.integers(-25, 25, size=noisy.shape)
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)

        denoised = denoise_image(noisy)
        assert denoised.std() < noisy.std()

    def test_preserves_shape_and_dtype(self, sample_image):
        result = denoise_image(sample_image)
        assert result.shape == sample_image.shape
        assert result.dtype == sample_image.dtype
