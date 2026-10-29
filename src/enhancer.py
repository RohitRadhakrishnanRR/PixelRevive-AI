"""Classical (non-learned) image restoration: inpaint, denoise, contrast, sharpen.

Each stage is independently toggleable and runs in a fixed order
(inpaint -> denoise -> contrast -> sharpen) regardless of which are enabled.
"""

import cv2

from .config import (
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    DENOISE_H,
    DENOISE_H_COLOR,
    DENOISE_SEARCH_WINDOW,
    DENOISE_TEMPLATE_WINDOW,
    INPAINT_KERNEL_SIZE,
    INPAINT_RADIUS,
    INPAINT_THRESHOLD,
    SHARPEN_KERNEL,
)


def detect_scratches(image, threshold=INPAINT_THRESHOLD, kernel_size=INPAINT_KERNEL_SIZE):
    """Detect thin scratches, cracks, and dust specks via Top-Hat and Black-Hat morphology.

    - Top-Hat isolates thin bright structures (light cracks/creases).
    - Black-Hat isolates thin dark structures (dark scratches).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    black_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    combined = cv2.add(top_hat, black_hat)
    _, mask = cv2.threshold(combined, threshold, 255, cv2.THRESH_BINARY)

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(mask, dilate_kernel, iterations=1)


def inpaint_image(image, inpaint_radius=INPAINT_RADIUS, threshold=INPAINT_THRESHOLD):
    """Inpaint detected physical scratches and creases using Fast Marching (Telea)."""
    mask = detect_scratches(image, threshold=threshold)
    if cv2.countNonZero(mask) == 0:
        return image
    return cv2.inpaint(image, mask, inpaint_radius, cv2.INPAINT_TELEA)


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


def enhance_image(image, inpaint=False, denoise=True, contrast=True, sharpen=True):
    """Run the classical restoration pipeline on a BGR uint8 image.

    Each stage can be independently enabled or disabled; disabled stages
    are skipped entirely rather than run with a no-op setting.
    """
    result = image

    if inpaint:
        result = inpaint_image(result)

    if denoise:
        result = denoise_image(result)

    if contrast:
        result = enhance_contrast(result)

    if sharpen:
        result = sharpen_image(result)

    return result

