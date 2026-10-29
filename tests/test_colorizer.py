"""Tests for src/colorizer.py.

These test the pipeline's shape/dtype/range handling and error behavior
using a stub network (see conftest.py) rather than the real ~123MB model,
which isn't available in this environment. They can't verify the model's
actual color predictions -- there's nothing meaningful to assert about
those without the real weights -- but they do verify the surrounding code
that every real inference also goes through.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

import src.colorizer as colorizer


class TestLoadModelMissingFiles:
    """load_model() must fail with a clear, actionable error instead of
    letting a raw OpenCV error crash the app (the P0 fix)."""

    def test_missing_model_file_raises_filenotfounderror(self, monkeypatch):
        monkeypatch.setattr(colorizer, "MODEL_PATH", Path("/tmp/does_not_exist.caffemodel"))
        with pytest.raises(FileNotFoundError, match="does_not_exist.caffemodel"):
            colorizer.load_model()

    def test_error_names_only_the_actually_missing_file(self, monkeypatch):
        monkeypatch.setattr(colorizer, "MODEL_PATH", Path("/tmp/does_not_exist.caffemodel"))
        with pytest.raises(FileNotFoundError) as excinfo:
            colorizer.load_model()
        # PROTOTXT_PATH and POINTS_PATH are real, present files -- only the
        # model weights should be reported as missing
        assert "colorization_deploy_v2.prototxt" not in str(excinfo.value)


class TestColorizeOutput:
    def test_output_dimensions_match_input(self, stub_net):
        image = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
        result = colorizer.colorize(image, stub_net)
        assert result.shape[:2] == image.shape[:2]

    def test_output_has_three_channels(self, stub_net, sample_image):
        result = colorizer.colorize(sample_image, stub_net)
        assert result.shape[2] == 3

    def test_output_is_in_valid_range_for_a_uint8_cast(self, stub_net, sample_image):
        result = colorizer.colorize(sample_image, stub_net)
        assert result.min() >= 0
        assert result.max() <= 1
        assert not np.isnan(result).any()

    def test_non_square_and_odd_dimensions_do_not_error(self, stub_net):
        # the network always resizes to 224x224 internally; the calling
        # image's own dimensions shouldn't matter, including odd ones
        image = np.random.randint(0, 255, (37, 51, 3), dtype=np.uint8)
        result = colorizer.colorize(image, stub_net)
        assert result.shape[:2] == (37, 51)


class TestColorizeClipRegression:
    """Regression test for a P0 bug: colorize() applied a fractional power
    to LAB->BGR output with no clip first. A fractional power of a
    negative number is NaN in NumPy.

    In this OpenCV build, cv2.cvtColor(..., COLOR_LAB2BGR) appears to
    already clamp float32 output to [0, 1] internally -- checked across
    200k random LAB points plus boundary extremes (L=0/100, |a|,|b| up to
    200) and found none outside range. So this test forces the condition
    via a patched cv2.cvtColor rather than relying on triggering it
    through real network output, to directly verify the defensive clip
    still does its job regardless of what any given OpenCV build does
    internally.
    """

    def test_out_of_gamut_pixel_does_not_produce_nan(self, stub_net, monkeypatch):
        real_cvtColor = cv2.cvtColor

        def rigged_cvtColor(src, code, *args, **kwargs):
            out = real_cvtColor(src, code, *args, **kwargs)
            if code == cv2.COLOR_LAB2BGR:
                out = out.copy()
                out[0, 0, 0] = -0.05
                out[0, 0, 1] = 1.2
            return out

        monkeypatch.setattr(cv2, "cvtColor", rigged_cvtColor)

        image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        result = colorizer.colorize(image, stub_net)

        assert not np.isnan(result).any()
        assert result.min() >= 0
        assert result.max() <= 1
