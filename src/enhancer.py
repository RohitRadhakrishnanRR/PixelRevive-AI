"""Classical (non-learned) image restoration: denoise, contrast, sharpen.

Each stage is independently toggleable and runs in a fixed order
(denoise -> contrast -> sharpen) regardless of which are enabled, matching
the pipeline order in the project README.
"""

import cv2

from .config import (
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    DENOISE_H,
    DENOISE_H_COLOR,
    DENOISE_SEARCH_WINDOW,
    DENOISE_TEMPLATE_WINDOW,
    SHARPEN_KERNEL,
)


def denoise_image(image):
    """Remove noise while preserving edges, via non-local means denoising."""
    return cv2.fastNlMeansDenoisingColored(
        image,
        None,
        DENOISE_H,
        DENOISE_H_COLOR,
        DENOISE_TEMPLATE_WINDOW,
        DENOISE_SEARCH_WINDOW
    )


def enhance_contrast(image):
    """Boost local contrast with CLAHE, applied to the L (lightness)
    channel in LAB space only, so color and saturation aren't affected."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE
    )
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def sharpen_image(image):
    """Sharpen with a unity-gain Laplacian kernel (sums to 1.0, see
    config.py), so flat regions keep their original brightness."""
    return cv2.filter2D(
        image,
        -1,
        SHARPEN_KERNEL
    )


def enhance_image(image, denoise=True, contrast=True, sharpen=True):
    """Run the classical restoration pipeline on a BGR uint8 image.

    Each stage can be independently enabled or disabled; disabled stages
    are skipped entirely rather than run with a no-op setting.
    """
    result = image

    if denoise:
        result = denoise_image(result)

    if contrast:
        result = enhance_contrast(result)

    if sharpen:
        result = sharpen_image(result)

    return result

