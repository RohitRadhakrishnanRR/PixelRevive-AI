"""Shared constants and file paths for the colorization pipeline.

Centralizing these avoids magic numbers scattered across the pipeline and
makes every tunable value easy to find and change in one place.
"""

from pathlib import Path

import numpy as np

# --- paths ------------------------------------------------------------
# Resolved relative to this file, not the process's working directory, so
# the app works no matter where it's launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model"

PROTOTXT_PATH = MODEL_DIR / "colorization_deploy_v2.prototxt"
MODEL_PATH = MODEL_DIR / "colorization_release_v2.caffemodel"
POINTS_PATH = MODEL_DIR / "pts_in_hull.npy"

# Official source (Richard Zhang / UC Berkeley), the same URL OpenCV's own
# sample code and documentation point to. Plain HTTP because that's the
# long-standing canonical link every reference uses; MIN_MODEL_SIZE_BYTES
# below is a basic integrity guard against a truncated or corrupted
# download, not a substitute for a real checksum.
MODEL_URL = "https://mirror.opencv.ai/colorization_release_v2.caffemodel"
MIN_MODEL_SIZE_BYTES = 100_000_000  # real file is ~123MB

# --- colorization network ----------------------------------------------
NETWORK_INPUT_SIZE = 224      # the model expects a 224x224 L channel
LAB_L_MEAN_OFFSET = 50        # mean-centers L before feeding the network
NUM_AB_CLUSTERS = 313         # quantized ab bins the model predicts over
CLUSTER_CENTER_BIAS = 2.606   # bias term for the cluster-rebalancing layer
SATURATION_GAMMA = 0.9        # slight saturation boost after LAB -> BGR

# --- denoising ----------------------------------------------------------
DENOISE_H = 8                 # filter strength, luminance
DENOISE_H_COLOR = 8           # filter strength, color channels
DENOISE_TEMPLATE_WINDOW = 7
DENOISE_SEARCH_WINDOW = 21

# --- sharpening -----------------------------------------------------
# Unity-gain Laplacian sharpen kernel (sums to 1.0 so flat regions keep
# their original brightness instead of being amplified).
SHARPEN_KERNEL = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

# --- contrast enhancement (CLAHE) ---------------------------------------
CLAHE_CLIP_LIMIT = 2.5        # higher = stronger local contrast, more noise risk
CLAHE_TILE_GRID_SIZE = (8, 8)

# --- input limits ---------------------------------------------------------
# Longest side, in pixels. cv2.fastNlMeansDenoisingColored is the slow part
# of the pipeline on CPU; this keeps processing time reasonable on a
# CPU-only deployment without visibly hurting typical restoration use cases.
MAX_IMAGE_DIMENSION = 1600

# Cap on how many images a single batch-processing run will accept, so a
# large upload can't hold unbounded memory or run for an unbounded time.
MAX_BATCH_FILES = 10

