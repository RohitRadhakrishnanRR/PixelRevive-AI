"""CNN-based image colorization using OpenCV's dnn module.

Loads the pretrained Zhang et al. (2016) colorization network and applies
it in LAB color space: the network predicts the A/B (color) channels from
the L (lightness) channel, which are then recombined and converted back to
BGR.
"""

import cv2
import numpy as np

from .config import (
    CLUSTER_CENTER_BIAS,
    LAB_L_MEAN_OFFSET,
    MODEL_PATH,
    NETWORK_INPUT_SIZE,
    NUM_AB_CLUSTERS,
    POINTS_PATH,
    PROTOTXT_PATH,
    SATURATION_GAMMA,
)
from .model_downloader import ensure_model_downloaded

def load_model():
    """Load the Caffe colorization network and wire in the cluster centers."""

    # Download the large pretrained weights if they are missing.
    if not MODEL_PATH.is_file():
        ensure_model_downloaded()

    missing = [p for p in (PROTOTXT_PATH, MODEL_PATH, POINTS_PATH) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing colorization model file(s): "
            + ", ".join(str(p) for p in missing)
            + ". See the README's 'Model Setup' section for download instructions."
        )

    # load neural network
    net = cv2.dnn.readNetFromCaffe(
        str(PROTOTXT_PATH),
        str(MODEL_PATH)
    )

    # load cluster centers
    pts = np.load(POINTS_PATH)

    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")

    pts = pts.transpose().reshape(2, NUM_AB_CLUSTERS, 1, 1)

    net.getLayer(class8).blobs = [
        pts.astype("float32")
    ]

    net.getLayer(conv8).blobs = [
        np.full([1, NUM_AB_CLUSTERS], CLUSTER_CENTER_BIAS, dtype="float32")
    ]

    return net


def colorize(image, net):
    """Colorize a BGR uint8 image using a loaded colorization network.

    `net` comes from `load_model()`. Splitting loading from inference lets
    the caller (the Streamlit app) own caching, and lets tests call this
    with a stub network instead of the real 123MB model.
    """

    # normalize image
    scaled = image.astype("float32") / 255.0

    # convert to LAB color space
    lab = cv2.cvtColor(
        scaled,
        cv2.COLOR_BGR2LAB
    )

    # extract L channel
    L = lab[:, :, 0]

    # resize for neural network
    L_resized = cv2.resize(
        L,
        (NETWORK_INPUT_SIZE, NETWORK_INPUT_SIZE)
    )

    L_resized -= LAB_L_MEAN_OFFSET

    # prediction
    net.setInput(
        cv2.dnn.blobFromImage(L_resized)
    )

    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # resize prediction back
    ab = cv2.resize(
        ab,
        (image.shape[1], image.shape[0])
    )

    # combine L + AB channels
    L = cv2.resize(
        L,
        (image.shape[1], image.shape[0])
    )

    colorized = np.concatenate(
        (L[:, :, np.newaxis], ab),
        axis=2
    )

    # convert LAB → BGR
    colorized = cv2.cvtColor(
        colorized,
        cv2.COLOR_LAB2BGR
    )

    # clip to a valid [0, 1] range: LAB → BGR can legitimately land
    # slightly outside this range since the predicted AB channels aren't
    # guaranteed to stay in-gamut for a given L. Skipping this turns into
    # NaN after the power below, then black speckle artifacts once the
    # result is cast to uint8.
    colorized = np.clip(colorized, 0, 1)

    # improve saturation slightly
    colorized = np.power(colorized, SATURATION_GAMMA)

    return colorized
