import io

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from src.colorizer import load_model
from src.config import MAX_BATCH_FILES, MAX_IMAGE_DIMENSION, MODEL_PATH
from src.image_utils import (
    build_comparison_html,
    build_results_zip,
    format_file_size,
    resize_if_too_large,
)
from src.metrics import estimate_contrast, estimate_sharpness, get_processing_device
from src.model_downloader import ensure_model_downloaded
from src.pipeline import run_pipeline

st.set_page_config(
    page_title="AI Photo Restoration Studio",
    page_icon="🎨",
    layout="wide"
)

st.markdown("""
# 🎨 AI Photo Restoration & Colorization Studio

Restore, enhance, and colorize photographs using computer vision and deep learning.
""")

# --- Sidebar: processing controls (shared by both tabs below) -----------
st.sidebar.header("⚙️ Processing")
st.sidebar.caption("Choose which steps to apply before colorizing.")

restoration_enabled = st.sidebar.checkbox("🛠️ Restoration", value=True)
denoise_enabled = st.sidebar.checkbox(
    "🧹 Denoising", value=True, disabled=not restoration_enabled
)
contrast_enabled = st.sidebar.checkbox(
    "🌗 Contrast Enhancement", value=True, disabled=not restoration_enabled
)
sharpen_enabled = st.sidebar.checkbox(
    "✨ Sharpening", value=True, disabled=not restoration_enabled
)

st.sidebar.divider()
colorize_enabled = st.sidebar.checkbox("🎨 AI Colorization", value=True)

denoise_flag = denoise_enabled and restoration_enabled
contrast_flag = contrast_enabled and restoration_enabled
sharpen_flag = sharpen_enabled and restoration_enabled


@st.cache_resource(show_spinner="Loading colorization model…")
def get_cached_model():
    return load_model()


def ensure_model_ready():
    """Downloads the model (with a progress bar) if colorization is on and
    the weights aren't present yet. Safe to call even when already present."""
    if colorize_enabled and not MODEL_PATH.is_file():
        progress_bar = st.progress(
            0.0,
            text="Downloading colorization model (first run only, ~123MB)…"
        )

        def _report_progress(downloaded, total):
            if total:
                progress_bar.progress(min(downloaded / total, 1.0))

        ensure_model_downloaded(progress_callback=_report_progress)
        progress_bar.empty()


def render_before_after(before_np, after_np, before_label, after_label):
    """Interactive drag-to-reveal comparison (see build_comparison_html)."""
    html, height = build_comparison_html(before_np, after_np, before_label, after_label)
    components.html(html, height=height)


def read_upload_as_rgb(uploaded_file):
    """Open an uploaded file, return (rgb_array, original_pil_mode)."""
    raw_image = Image.open(uploaded_file)
    original_mode = raw_image.mode
    rgb = np.array(raw_image.convert("RGB"))
    return rgb, original_mode


def encode_png(image_rgb):
    buf = io.BytesIO()
    Image.fromarray(image_rgb).save(buf, format="PNG")
    return buf.getvalue()


tab_single, tab_batch = st.tabs(["🖼️ Single Image", "📚 Batch Processing"])

# ===========================================================================
# Single image
# ===========================================================================
with tab_single:

    with st.container(border=True):
        st.markdown("### 📤 Upload a Photo")
        uploaded = st.file_uploader(
            "Drag and drop or browse — JPG, PNG, or WEBP",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
            key="single_uploader"
        )

    if uploaded is not None:

        try:
            image_np, original_mode = read_upload_as_rgb(uploaded)
        except Exception as e:
            st.error(f"Couldn't read this file as an image: {e}")
            st.stop()

        image_np, was_resized = resize_if_too_large(image_np)
        # image_np (RGB) is what actually gets processed from here on, so
        # it's also the correct "before" reference for the slider below --
        # it must match the result's dimensions exactly for the overlay to
        # align.

        st.caption(
            f"**{uploaded.name}** · {image_np.shape[1]}×{image_np.shape[0]}px · "
            f"{format_file_size(uploaded.size)} · {original_mode} mode"
        )
        if was_resized:
            st.info(
                f"Downscaled to {image_np.shape[1]}×{image_np.shape[0]} for "
                f"processing (limit: {MAX_IMAGE_DIMENSION}px on the longest side)."
            )

        with st.container(border=True):
            st.markdown("### 📷 Preview")
            st.image(image_np, use_container_width=True)
            run = st.button("✨ Restore & Colorize", type="primary")

        if run:
            try:
                ensure_model_ready()
                with st.spinner("AI is restoring and colorizing image..."):
                    net = get_cached_model() if colorize_enabled else None
                    result, elapsed_seconds = run_pipeline(
                        image_np, net,
                        denoise=denoise_flag, contrast=contrast_flag, sharpen=sharpen_flag,
                        colorize_enabled=colorize_enabled
                    )
            except FileNotFoundError as e:
                st.error(f"Colorization model isn't available: {e}")
            except Exception as e:
                st.error(f"Something went wrong while processing this image: {e}")
            else:
                with st.container(border=True):

                    result_heading = "🌈 Restored & Colorized" if colorize_enabled else "🛠️ Restored"
                    st.markdown(f"### {result_heading}")

                    render_before_after(image_np, result, "Original", "Result")

                    st.divider()

                    sharpness_after = estimate_sharpness(result)
                    contrast_after = estimate_contrast(result)
                    sharpness_delta = sharpness_after - estimate_sharpness(image_np)
                    contrast_delta = contrast_after - estimate_contrast(image_np)

                    metric_cols = st.columns(5)
                    metric_cols[0].metric("Resolution", f"{result.shape[1]} × {result.shape[0]}")
                    metric_cols[1].metric("Processing Time", f"{elapsed_seconds:.2f} s")
                    metric_cols[2].metric("Sharpness*", f"{sharpness_after:.1f}", delta=f"{sharpness_delta:+.1f}")
                    metric_cols[3].metric("Contrast*", f"{contrast_after:.1f}", delta=f"{contrast_delta:+.1f}")
                    metric_cols[4].metric("Device", get_processing_device())

                    st.caption(
                        "*Heuristic estimates (Laplacian variance for sharpness, pixel "
                        "intensity std. dev. for contrast) — not calibrated quality scores."
                    )

                    download_name = "restored_colorized.png" if colorize_enabled else "restored.png"
                    st.download_button(
                        label="📥 Download Result",
                        data=encode_png(result),
                        file_name=download_name,
                        mime="image/png"
                    )

                    st.success("Done!")

# ===========================================================================
# Batch processing
# ===========================================================================
with tab_batch:

    with st.container(border=True):
        st.markdown("### 📤 Upload Multiple Photos")
        batch_files = st.file_uploader(
            f"Up to {MAX_BATCH_FILES} images — JPG, PNG, or WEBP",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="batch_uploader"
        )

    if batch_files:

        if len(batch_files) > MAX_BATCH_FILES:
            st.warning(
                f"{len(batch_files)} images selected — only the first "
                f"{MAX_BATCH_FILES} will be processed to keep memory and "
                f"processing time bounded."
            )
            batch_files = batch_files[:MAX_BATCH_FILES]

        st.caption(f"{len(batch_files)} image(s) ready.")
        process_all = st.button("✨ Process All", type="primary")

        if process_all:
            try:
                ensure_model_ready()
            except Exception:
                pass  # a per-file FileNotFoundError below will report this clearly

            progress_label = st.empty()
            progress_bar = st.progress(0.0)
            net = get_cached_model() if colorize_enabled else None

            results = []  # (filename, png_bytes or None, error or None)
            for i, file in enumerate(batch_files):
                progress_label.text(f"{i} / {len(batch_files)} processed")
                try:
                    image_np, _mode = read_upload_as_rgb(file)
                    image_np, _resized = resize_if_too_large(image_np)
                    result, _elapsed = run_pipeline(
                        image_np, net,
                        denoise=denoise_flag, contrast=contrast_flag, sharpen=sharpen_flag,
                        colorize_enabled=colorize_enabled
                    )
                    results.append((file.name, encode_png(result), None))
                except Exception as e:
                    results.append((file.name, None, str(e)))
                progress_bar.progress((i + 1) / len(batch_files))

            progress_label.text(f"{len(batch_files)} / {len(batch_files)} processed")

            succeeded = [(name, data) for name, data, err in results if err is None]
            failed = [(name, err) for name, data, err in results if err is not None]

            with st.container(border=True):
                st.markdown("### Results")
                st.caption(f"{len(succeeded)} succeeded, {len(failed)} failed.")

                if succeeded:
                    st.download_button(
                        "📥 Download All as ZIP",
                        data=build_results_zip([
                            (f"restored_{name}" if not name.lower().endswith(".png") else f"restored_{name[:-4]}.png", data)
                            for name, data in succeeded
                        ]),
                        file_name="restored_batch.zip",
                        mime="application/zip"
                    )

                cols = st.columns(3)
                for idx, (name, data) in enumerate(succeeded):
                    with cols[idx % 3]:
                        st.image(data, caption=name, use_container_width=True)
                        st.download_button(
                            "Download", data=data, file_name=f"restored_{name}",
                            mime="image/png", key=f"dl_{idx}_{name}"
                        )

                for name, err in failed:
                    st.error(f"**{name}**: {err}")
