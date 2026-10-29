"""Tests for src/image_utils.py."""

import base64
import io

import numpy as np
import pytest
from PIL import Image

from src.image_utils import (
    build_comparison_html,
    build_results_zip,
    format_file_size,
    image_to_base64_png,
    resize_if_too_large,
)


class TestResizeIfTooLarge:
    def test_small_image_passes_through_unchanged(self):
        image = np.zeros((800, 600, 3), dtype=np.uint8)
        result, was_resized = resize_if_too_large(image, max_dimension=1600)
        assert was_resized is False
        assert result.shape == image.shape
        assert result is image  # no copy needed when nothing changes

    def test_large_image_is_downscaled_to_the_limit(self):
        image = np.zeros((3000, 2000, 3), dtype=np.uint8)
        result, was_resized = resize_if_too_large(image, max_dimension=1600)
        assert was_resized is True
        assert max(result.shape[:2]) == 1600

    def test_aspect_ratio_is_preserved(self):
        image = np.zeros((3000, 2000, 3), dtype=np.uint8)  # 3:2
        result, _ = resize_if_too_large(image, max_dimension=1600)
        original_ratio = 3000 / 2000
        new_ratio = result.shape[0] / result.shape[1]
        assert abs(original_ratio - new_ratio) < 0.01

    def test_image_exactly_at_the_limit_is_not_resized(self):
        image = np.zeros((1600, 1200, 3), dtype=np.uint8)
        result, was_resized = resize_if_too_large(image, max_dimension=1600)
        assert was_resized is False


class TestBase64Encoding:
    def test_roundtrips_to_identical_pixels(self):
        image = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
        uri = image_to_base64_png(image)
        assert uri.startswith("data:image/png;base64,")

        raw = base64.b64decode(uri.split(",", 1)[1])
        decoded = np.array(Image.open(io.BytesIO(raw)))
        assert np.array_equal(image, decoded)


class TestComparisonHtmlBuilder:
    def test_no_leftover_placeholders(self):
        before = np.zeros((100, 200, 3), dtype=np.uint8)
        after = np.full((100, 200, 3), 255, dtype=np.uint8)
        html, _ = build_comparison_html(before, after, "Original", "Result")
        for token in ("__WIDTH__", "__BEFORE_URI__", "__AFTER_URI__", "__BEFORE_LABEL__", "__AFTER_LABEL__"):
            assert token not in html

    def test_labels_are_embedded(self):
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        html, _ = build_comparison_html(image, image, "Before Label", "After Label")
        assert "Before Label" in html
        assert "After Label" in html

    def test_both_images_are_embedded(self):
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        html, _ = build_comparison_html(image, image, "A", "B")
        assert html.count("data:image/png;base64,") == 2

    def test_height_matches_aspect_ratio(self):
        # 2:1 image (height:width) at a given display width
        before = np.zeros((400, 800, 3), dtype=np.uint8)
        after = np.zeros((400, 800, 3), dtype=np.uint8)
        _, height = build_comparison_html(before, after, "A", "B", display_width=680)
        assert height == round(680 * 400 / 800) + 10

    def test_mismatched_image_sizes_raise(self):
        # the overlay technique requires both images to be the same size;
        # silently misaligning them would be worse than an explicit error
        before = np.zeros((100, 100, 3), dtype=np.uint8)
        after = np.zeros((50, 50, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="same size"):
            build_comparison_html(before, after, "A", "B")


class TestFormatFileSize:
    @pytest.mark.parametrize("num_bytes,expected", [
        (500, "500 B"),
        (2048, "2.0 KB"),
        (5_000_000, "4.8 MB"),
        (3_000_000_000, "2.8 GB"),
    ])
    def test_known_magnitudes(self, num_bytes, expected):
        assert format_file_size(num_bytes) == expected


class TestBuildResultsZip:
    def test_contains_all_files_with_correct_content(self):
        import io
        import zipfile

        zip_bytes = build_results_zip([
            ("a.png", b"fake-png-bytes-a"),
            ("b.png", b"fake-png-bytes-b"),
        ])
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            assert set(zf.namelist()) == {"a.png", "b.png"}
            assert zf.read("a.png") == b"fake-png-bytes-a"
            assert zf.read("b.png") == b"fake-png-bytes-b"

    def test_empty_list_produces_a_valid_empty_zip(self):
        import io
        import zipfile

        zip_bytes = build_results_zip([])
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            assert zf.namelist() == []
