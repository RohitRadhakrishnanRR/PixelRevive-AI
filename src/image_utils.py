"""Utilities for preparing images before they enter the processing pipeline,
and for presenting them in the UI."""

import base64
import io
import zipfile

import cv2
from PIL import Image

from .config import MAX_IMAGE_DIMENSION


def resize_if_too_large(image, max_dimension=MAX_IMAGE_DIMENSION):
    """Downscale an image if its longer side exceeds max_dimension.

    Preserves aspect ratio and only touches images that actually exceed the
    limit. Returns (image, was_resized) so callers can tell the user their
    image was downscaled for processing.
    """
    height, width = image.shape[:2]
    longest_side = max(height, width)

    if longest_side <= max_dimension:
        return image, False

    scale = max_dimension / longest_side
    new_size = (round(width * scale), round(height * scale))

    # INTER_AREA gives the best quality when shrinking an image
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized, True


def image_to_base64_png(image_array):
    """Encode an RGB uint8 numpy array as a base64 PNG data URI."""
    pil_image = Image.fromarray(image_array)
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


_COMPARISON_HTML_TEMPLATE = """
<div class="ba-wrap" style="max-width: __WIDTH__px;">
  <img class="ba-img" src="__BEFORE_URI__">
  <img class="ba-img ba-after" id="ba-after" src="__AFTER_URI__">
  <div class="ba-divider" id="ba-divider"></div>
  <span class="ba-tag ba-tag-left">__BEFORE_LABEL__</span>
  <span class="ba-tag ba-tag-right">__AFTER_LABEL__</span>
  <input type="range" min="0" max="100" value="50" class="ba-range" id="ba-range">
</div>
<style>
  .ba-wrap { position: relative; width: 100%; margin: 0 auto; border-radius: 10px; overflow: hidden; line-height: 0; }
  .ba-img { display: block; width: 100%; height: auto; }
  .ba-after { position: absolute; top: 0; left: 0; clip-path: inset(0 50% 0 0); }
  .ba-divider { position: absolute; top: 0; bottom: 0; left: 50%; width: 3px; background: #fff; box-shadow: 0 0 6px rgba(0,0,0,.6); pointer-events: none; }
  .ba-range { position: absolute; top: 0; left: 0; width: 100%; height: 100%; margin: 0; opacity: 0; cursor: ew-resize; }
  .ba-tag { position: absolute; top: 10px; padding: 3px 10px; border-radius: 999px; background: rgba(0,0,0,.6); color: #fff; font: 500 12px sans-serif; pointer-events: none; }
  .ba-tag-left { left: 10px; }
  .ba-tag-right { right: 10px; }
</style>
<script>
  const range = document.getElementById('ba-range');
  const after = document.getElementById('ba-after');
  const divider = document.getElementById('ba-divider');
  range.addEventListener('input', function (e) {
    const v = e.target.value;
    after.style.clipPath = 'inset(0 ' + (100 - v) + '% 0 0)';
    divider.style.left = v + '%';
  });
</script>
"""


def build_comparison_html(before_np, after_np, before_label="Original", after_label="Result", display_width=680):
    """Build the HTML/CSS/JS for an interactive drag-to-reveal comparison.

    Returns (html, height); height is what the caller passes to
    st.components.v1.html() so the iframe fits without scrollbars.

    before_np and after_np must be the same size -- they're stacked with a
    CSS clip-path, not cropped to match, so mismatched sizes misalign.
    """
    if before_np.shape[:2] != after_np.shape[:2]:
        raise ValueError(
            f"before/after images must be the same size to align correctly, "
            f"got {before_np.shape[:2]} and {after_np.shape[:2]}"
        )

    height, width = before_np.shape[:2]
    display_height = round(display_width * height / width)

    html = (
        _COMPARISON_HTML_TEMPLATE
        .replace("__WIDTH__", str(display_width))
        .replace("__BEFORE_URI__", image_to_base64_png(before_np))
        .replace("__AFTER_URI__", image_to_base64_png(after_np))
        .replace("__BEFORE_LABEL__", before_label)
        .replace("__AFTER_LABEL__", after_label)
    )
    return html, display_height + 10


def format_file_size(num_bytes):
    """Human-readable file size, e.g. 1536 -> '1.5 KB'."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024


def build_results_zip(named_png_bytes):
    """Bundle (filename, png_bytes) pairs into one in-memory ZIP.

    Returns the ZIP's raw bytes, ready to hand to a download button.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename, data in named_png_bytes:
            zf.writestr(filename, data)
    return buf.getvalue()


