"""Shared pytest fixtures."""

import numpy as np
import pytest


@pytest.fixture
def sample_image():
    """A small synthetic uint8 image with real structure -- a brightness
    gradient, a hard edge, and mild noise -- rather than flat noise, so
    tests exercise something closer to a real photo."""
    rng = np.random.default_rng(42)
    height, width = 64, 96
    base = np.tile(np.linspace(40, 200, width, dtype=np.uint8), (height, 1))
    image = np.stack([base, base, base], axis=-1).astype(np.int16)
    image += rng.integers(-10, 10, size=image.shape)
    image[:, width // 2:] += 30
    return np.clip(image, 0, 255).astype(np.uint8)


@pytest.fixture
def flat_image():
    """A flat mid-gray image, for brightness-preservation checks."""
    return np.full((50, 50, 3), 150, dtype=np.uint8)


class StubColorizationNet:
    """A minimal stand-in for the real cv2.dnn network, so colorize() can
    be tested without the real ~123MB model file. Always predicts all-zero
    AB channels -- enough to exercise the pipeline's shape/dtype/range
    handling, not to test the model's actual color predictions (there's
    nothing to meaningfully assert about those without the real weights)."""

    def setInput(self, blob):
        pass

    def forward(self):
        return np.zeros((1, 2, 224, 224), dtype=np.float32)


@pytest.fixture
def stub_net():
    return StubColorizationNet()
