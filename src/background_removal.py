
import cv2
import numpy as np


def remove_background(image_rgb, rect, iterations=5):
    """
    Remove background using OpenCV GrabCut.

    image_rgb: Original RGB image
    rect: (x, y, width, height)
    iterations: GrabCut iterations

    Returns an RGBA image with transparent background.
    """

    if image_rgb is None or image_rgb.size == 0:
        raise ValueError("Invalid input image.")

    height, width = image_rgb.shape[:2]

    x, y, rect_width, rect_height = map(int, rect)

    # Validate rectangle
    if (
        x < 0 or y < 0
        or rect_width <= 1 or rect_height <= 1
        or x + rect_width > width
        or y + rect_height > height
    ):
        raise ValueError("Invalid rectangle selection.")

    # Convert RGB to BGR
    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR
    )

    # Initialize GrabCut mask and models
    mask = np.zeros((height, width), np.uint8)

    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    # Apply GrabCut
    cv2.grabCut(
        image_bgr,
        mask,
        (x, y, rect_width, rect_height),
        bg_model,
        fg_model,
        int(iterations),
        cv2.GC_INIT_WITH_RECT
    )

    # Extract foreground
    foreground_mask = np.where(
        (mask == cv2.GC_FGD) |
        (mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype(np.uint8)

    # Refine mask
    kernel = np.ones((3, 3), np.uint8)

    foreground_mask = cv2.morphologyEx(
        foreground_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Smooth boundaries
    foreground_mask = cv2.GaussianBlur(
        foreground_mask,
        (3, 3),
        0
    )

    # Create transparent RGBA output
    result = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2RGBA
    )

    result[:, :, 3] = foreground_mask

    return result
