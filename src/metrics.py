"""Heuristic image-quality metrics and processing device detection.

These are simple, well-established CV heuristics -- Laplacian variance for
sharpness, pixel standard deviation for contrast, mean intensity for
brightness -- not calibrated quality scores, and not a substitute for
PSNR/SSIM against a real reference image. They're useful for eyeballing
how an image changed through the pipeline, not for absolute quality
claims. All three expect an RGB image (2D grayscale or 3D color).
"""

import cv2
import numpy as np


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


def estimate_colorfulness(image):
    """Hasler and Süsstrunk (2003) colorfulness index.

    Monochrome / grayscale images score 0.0, while colorized images
    exhibit positive scores proportional to chromatic vibrancy.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        return 0.0

    R = image[:, :, 0].astype("float32")
    G = image[:, :, 1].astype("float32")
    B = image[:, :, 2].astype("float32")

    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)

    std_rg = float(np.std(rg))
    std_yb = float(np.std(yb))
    mean_rg = float(np.mean(rg))
    mean_yb = float(np.mean(yb))

    std_rgyb = float(np.sqrt(std_rg ** 2 + std_yb ** 2))
    mean_rgyb = float(np.sqrt(mean_rg ** 2 + mean_yb ** 2))

    return float(std_rgyb + 0.3 * mean_rgyb)


def compute_intensity_histogram(image, bins=32):
    """Compute normalized histogram of luminance values (0-255)."""
    gray = _to_gray(image)
    hist, _ = np.histogram(gray, bins=bins, range=(0, 256))
    return (hist / float(gray.size)).tolist()


def compute_rgb_histogram(image, bins=32):
    """Compute normalized histogram for Red, Green, and Blue channels."""
    if image.ndim != 3:
        mono = compute_intensity_histogram(image, bins=bins)
        return {"Red": mono, "Green": mono, "Blue": mono}

    total = float(image.shape[0] * image.shape[1])
    r_hist, _ = np.histogram(image[:, :, 0], bins=bins, range=(0, 256))
    g_hist, _ = np.histogram(image[:, :, 1], bins=bins, range=(0, 256))
    b_hist, _ = np.histogram(image[:, :, 2], bins=bins, range=(0, 256))

    return {
        "Red": (r_hist / total).tolist(),
        "Green": (g_hist / total).tolist(),
        "Blue": (b_hist / total).tolist(),
    }


def compute_dynamic_range(image):
    """Compute minimum, maximum, and span (dynamic range) of pixel intensities."""
    gray = _to_gray(image)
    min_val, max_val = int(gray.min()), int(gray.max())
    return min_val, max_val, max_val - min_val


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
