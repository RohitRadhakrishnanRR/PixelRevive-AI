import cv2
import numpy as np


def remove_object(image_rgb, mask, radius=5, method="Telea"):
    """
    Remove selected objects or scratches from an RGB image.

    image_rgb: Original RGB image
    mask: Binary mask (white = area to remove)
    radius: Inpainting radius
    method: Telea or Navier-Stokes
    """

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    mask = np.asarray(mask, dtype=np.uint8)
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)

    if method == "Telea":
        algorithm = cv2.INPAINT_TELEA
    else:
        algorithm = cv2.INPAINT_NS

    restored = cv2.inpaint(
        image_bgr,
        mask,
        float(radius),
        algorithm
    )

    return cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)