"""Tests for src/pipeline.py."""

import numpy as np

from src.pipeline import run_pipeline


class TestRunPipeline:
    def test_colorize_disabled_does_not_need_a_net(self, sample_image):
        result, elapsed = run_pipeline(sample_image, net=None, colorize_enabled=False)
        assert result.shape == sample_image.shape
        assert elapsed >= 0

    def test_colorize_enabled_uses_the_given_net(self, sample_image, stub_net):
        result, elapsed = run_pipeline(sample_image, net=stub_net, colorize_enabled=True)
        assert result.shape[:2] == sample_image.shape[:2]
        assert result.shape[2] == 3
        assert elapsed >= 0

    def test_all_restoration_stages_disabled_still_runs(self, sample_image, stub_net):
        # denoise/contrast/sharpen all off -- only colorization should apply
        result, _ = run_pipeline(
            sample_image, net=stub_net,
            denoise=False, contrast=False, sharpen=False, colorize_enabled=True
        )
        assert result.shape[:2] == sample_image.shape[:2]

    def test_output_is_uint8(self, sample_image, stub_net):
        result, _ = run_pipeline(sample_image, net=stub_net)
        assert result.dtype == np.uint8

    def test_elapsed_time_is_measured(self, sample_image):
        _, elapsed = run_pipeline(sample_image, net=None, colorize_enabled=False)
        assert isinstance(elapsed, float)
        assert elapsed < 30  # sane upper bound for a tiny test image
