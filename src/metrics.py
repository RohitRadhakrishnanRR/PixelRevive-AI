"""Heuristic image-quality metrics and processing device detection.

These are simple, well-established CV heuristics -- Laplacian variance for
sharpness, pixel standard deviation for contrast, mean intensity for
brightness -- not calibrated quality scores, and not a substitute for
PSNR/SSIM against a real reference image. They're useful for eyeballing
how an image changed through the pipeline, not for absolute quality
claims. All three expect an RGB image (2D grayscale or 3D color).
"""

import cv2


def _to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image


def estimate_sharpness(image):
    """Variance of the Laplacian: a standard, simple blur/sharpness proxy.
    Higher means more high-frequency detail/edges."""
    return float(cv2.Laplacian(_to_gray(image), cv2.CV_64F).var())


def estimate_contrast(image):
    """Standard deviation of pixel intensity: a simple global-contrast proxy."""
    return float(_to_gray(image).std())


def estimate_brightness(image):
    """Mean pixel intensity."""
    return float(_to_gray(image).mean())


def get_processing_device():
    """'CUDA' only if OpenCV was built with CUDA support AND a device is
    actually present; 'CPU' otherwise -- never claims CUDA when it isn't
    there. The opencv-python-headless build in requirements.txt is a
    CPU-only wheel, so this correctly reports 'CPU' in a standard install;
    that's expected, not a bug.
    """
    try:
        return "CUDA" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "CPU"
    except (AttributeError, cv2.error):
        return "CPU"
