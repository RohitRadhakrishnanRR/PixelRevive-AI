
import cv2
import numpy as np


def detect_edges(image_rgb, method="Canny",
                 threshold1=100, threshold2=200,
                 kernel_size=3):

    """
    Perform edge detection using different algorithms.

    image_rgb: Input RGB image
    method: Sobel, Prewitt, Canny or Laplacian
    threshold1, threshold2: Canny thresholds
    kernel_size: Kernel size for Sobel and Laplacian

    Returns a grayscale edge image.
    """

    # Convert RGB to grayscale
    gray = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2GRAY
    )

    # Remove small noise before edge detection
    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    if method == "Sobel":

        # Calculate horizontal and vertical gradients
        gx = cv2.Sobel(
            gray,
            cv2.CV_64F,
            1, 0,
            ksize=kernel_size
        )

        gy = cv2.Sobel(
            gray,
            cv2.CV_64F,
            0, 1,
            ksize=kernel_size
        )

        # Combine gradients
        magnitude = np.sqrt(gx ** 2 + gy ** 2)

        edges = cv2.normalize(
            magnitude,
            None,
            0, 255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

    elif method == "Prewitt":

        # Prewitt horizontal and vertical kernels
        kernel_x = np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ], dtype=np.float32)

        kernel_y = np.array([
            [1, 1, 1],
            [0, 0, 0],
            [-1, -1, -1]
        ], dtype=np.float32)

        gx = cv2.filter2D(
            gray,
            cv2.CV_32F,
            kernel_x
        )

        gy = cv2.filter2D(
            gray,
            cv2.CV_32F,
            kernel_y
        )

        magnitude = cv2.magnitude(gx, gy)

        edges = cv2.normalize(
            magnitude,
            None,
            0, 255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

    elif method == "Canny":

        edges = cv2.Canny(
            gray,
            threshold1,
            threshold2
        )

    elif method == "Laplacian":

        laplacian = cv2.Laplacian(
            gray,
            cv2.CV_64F,
            ksize=kernel_size
        )

        edges = cv2.convertScaleAbs(laplacian)

    else:
        raise ValueError("Unsupported edge detection method.")

    return edges
