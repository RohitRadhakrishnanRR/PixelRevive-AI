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
    page_title="NeuralRestore AI — Photo Restoration & Colorization",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Custom Design System & Glassmorphism Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

/* Ambient glow & dark theme background */
.stApp {
    background-color: #0b0f19;
    background-image: 
        radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.12) 0px, transparent 50%),
        radial-gradient(at 100% 0%, rgba(168, 85, 247, 0.12) 0px, transparent 50%),
        radial-gradient(at 50% 100%, rgba(14, 165, 233, 0.08) 0px, transparent 50%);
}

/* Hero Section */
.hero-wrapper {
    padding: 1.8rem 2.2rem 1.6rem 2.2rem;
    margin-bottom: 1.8rem;
    background: linear-gradient(135deg, rgba(23, 31, 51, 0.75) 0%, rgba(15, 23, 42, 0.85) 100%);
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(16px);
}

.hero-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 5px 14px;
    border-radius: 9999px;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(129, 140, 248, 0.35);
    color: #a5b4fc;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-bottom: 0.85rem;
}

.hero-title {
    font-size: 2.35rem;
    font-weight: 800;
    line-height: 1.2;
    margin: 0 0 0.5rem 0;
    color: #ffffff;
}

.hero-gradient-text {
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-desc {
    color: #94a3b8;
    font-size: 0.98rem;
    line-height: 1.6;
    max-width: 860px;
    margin: 0 0 1.15rem 0;
}

.hero-tags-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.hero-tech-badge {
    background: rgba(15, 23, 42, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.08);
    color: #cbd5e1;
    font-size: 0.76rem;
    padding: 4px 11px;
    border-radius: 8px;
    font-weight: 500;
    display: inline-flex;
    align-items: center;
    gap: 5px;
}

/* Glassmorphism Containers */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(17, 24, 39, 0.65) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 18px !important;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3) !important;
    backdrop-filter: blur(12px) !important;
    padding: 1rem !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #0c1220 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
}

.sidebar-header-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.6), rgba(15, 23, 42, 0.8));
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 1rem;
    margin-bottom: 1.2rem;
}

.sidebar-title {
    font-size: 0.88rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: #a5b4fc;
    margin-bottom: 0.35rem;
}

.sidebar-desc {
    font-size: 0.8rem;
    color: #94a3b8;
    line-height: 1.45;
}

.sidebar-arch-card {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 14px;
    margin-top: 1.4rem;
}

.sidebar-arch-title {
    font-size: 0.72rem;
    font-weight: 700;
    color: #818cf8;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    margin-bottom: 9px;
}

.sidebar-arch-row {
    font-size: 0.78rem;
    color: #94a3b8;
    display: flex;
    justify-content: space-between;
    margin-bottom: 6px;
}

.sidebar-arch-val {
    font-weight: 600;
    color: #e2e8f0;
}

/* Custom Primary Button with Shimmer */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #c026d3 100%) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.68rem 1.6rem !important;
    font-size: 0.95rem !important;
    box-shadow: 0 4px 22px rgba(124, 58, 237, 0.42) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 28px rgba(124, 58, 237, 0.65) !important;
    filter: brightness(1.08) !important;
}

/* Streamlit Metrics */
[data-testid="stMetric"] {
    background: rgba(19, 27, 46, 0.85) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-top: 3px solid #6366f1 !important;
    border-radius: 14px !important;
    padding: 0.95rem !important;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25) !important;
    transition: transform 0.2s ease !important;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-2px) !important;
}

[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
}

[data-testid="stMetricValue"] {
    color: #f8fafc !important;
    font-weight: 700 !important;
}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px !important;
    background: rgba(15, 23, 42, 0.6) !important;
    padding: 6px !important;
    border-radius: 14px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    color: #94a3b8 !important;
    font-weight: 600 !important;
    padding: 8px 20px !important;
    transition: all 0.2s ease !important;
}

.stTabs [aria-selected="true"] {
    background: rgba(99, 102, 241, 0.22) !important;
    color: #ffffff !important;
    border-bottom: 2px solid #818cf8 !important;
}

/* Metadata Chips */
.meta-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 8px 0 14px 0;
}

.meta-chip {
    font-size: 0.76rem;
    padding: 3px 10px;
    border-radius: 9999px;
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    color: #cbd5e1;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Hero Header Banner
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-wrapper">
  <div class="hero-pill">✨ Computer Vision & Deep Learning Studio</div>
  <h1 class="hero-title">NeuralRestore <span class="hero-gradient-text">& Colorize</span></h1>
  <p class="hero-desc">
    High-fidelity photographic restoration and chromatic synthesis powered by edge-preserving 
    spatial filters and deep convolutional neural inference in CIE L*a*b* color space.
  </p>
  <div class="hero-tags-row">
    <span class="hero-tech-badge">🧹 Non-Local Means Denoising</span>
    <span class="hero-tech-badge">🌗 L-Channel CLAHE Equalization</span>
    <span class="hero-tech-badge">✨ Laplacian Edge Sharpening</span>
    <span class="hero-tech-badge">🧠 Zhang et al. (ECCV) Colorizer</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar: Processing Controls & Engine Diagnostics
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header-card">
      <div class="sidebar-title">⚙️ Processing Engine</div>
      <div class="sidebar-desc">Configure classical restoration filters and neural color synthesis stages.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### 🛠️ Classical Enhancement")
    restoration_enabled = st.checkbox("Enable Restoration Filters", value=True)
    denoise_enabled = st.checkbox(
        "🧹 Denoising (NL-Means)", value=True, disabled=not restoration_enabled
    )
    contrast_enabled = st.checkbox(
        "🌗 Contrast (LAB CLAHE)", value=True, disabled=not restoration_enabled
    )
    sharpen_enabled = st.checkbox(
        "✨ Sharpening (Laplacian)", value=True, disabled=not restoration_enabled
    )

    st.divider()

    st.markdown("##### 🎨 Neural Colorization")
    colorize_enabled = st.checkbox("🧠 Deep Colorizer (Zhang et al.)", value=True)

    denoise_flag = denoise_enabled and restoration_enabled
    contrast_flag = contrast_enabled and restoration_enabled
    sharpen_flag = sharpen_enabled and restoration_enabled

    current_device = get_processing_device()
    device_color = "#34d399" if current_device == "CUDA" else "#38bdf8"

    st.markdown(f"""
    <div class="sidebar-arch-card">
      <div class="sidebar-arch-title">Engine Architecture</div>
      <div class="sidebar-arch-row">
        <span>Framework:</span>
        <span class="sidebar-arch-val">OpenCV DNN</span>
      </div>
      <div class="sidebar-arch-row">
        <span>Color Space:</span>
        <span class="sidebar-arch-val">CIE L*a*b*</span>
      </div>
      <div class="sidebar-arch-row">
        <span>Compute Unit:</span>
        <span style="font-weight: 700; color: {device_color};">{current_device}</span>
      </div>
      <div class="sidebar-arch-row">
        <span>AB Clusters:</span>
        <span class="sidebar-arch-val">313 Bins</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading colorization neural network…")
def get_cached_model():
    return load_model()


def ensure_model_ready():
    """Downloads the model weights if colorization is enabled and weights are missing."""
    if colorize_enabled and not MODEL_PATH.is_file():
        progress_bar = st.progress(
            0.0,
            text="Downloading colorization model weights (~123MB, first run only)…"
        )

        def _report_progress(downloaded, total):
            if total:
                progress_bar.progress(min(downloaded / total, 1.0))

        ensure_model_downloaded(progress_callback=_report_progress)
        progress_bar.empty()


def render_before_after(before_np, after_np, before_label, after_label):
    """Interactive drag-to-reveal comparison."""
    html, height = build_comparison_html(before_np, after_np, before_label, after_label)
    components.html(html, height=height)


def read_upload_as_rgb(uploaded_file):
    """Open an uploaded file and return (rgb_array, original_pil_mode)."""
    raw_image = Image.open(uploaded_file)
    original_mode = raw_image.mode
    rgb = np.array(raw_image.convert("RGB"))
    return rgb, original_mode


def encode_png(image_rgb):
    buf = io.BytesIO()
    Image.fromarray(image_rgb).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------------------------
tab_single, tab_batch = st.tabs(["🖼️ Single Photo Studio", "📚 Batch Processing Studio"])

# ===========================================================================
# 1. Single Image Studio
# ===========================================================================
with tab_single:
    with st.container(border=True):
        st.markdown("#### 📤 Upload Photograph")
        uploaded = st.file_uploader(
            "Select an image to restore and colorize (JPG, PNG, or WEBP)",
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

        st.markdown(f"""
        <div class="meta-chip-row">
          <span class="meta-chip">📄 <strong>{uploaded.name}</strong></span>
          <span class="meta-chip">📐 {image_np.shape[1]} × {image_np.shape[0]} px</span>
          <span class="meta-chip">💾 {format_file_size(uploaded.size)}</span>
          <span class="meta-chip">🎨 {original_mode} mode</span>
        </div>
        """, unsafe_allow_html=True)

        if was_resized:
            st.info(
                f"ℹ️ Image downscaled to {image_np.shape[1]}×{image_np.shape[0]} px "
                f"to optimize compute time (maximum threshold: {MAX_IMAGE_DIMENSION}px)."
            )

        with st.container(border=True):
            st.markdown("#### 📷 Input Preview")
            st.image(image_np, use_container_width=True)
            run = st.button("✨ Restore & Colorize Photo", type="primary")

        if run:
            try:
                ensure_model_ready()
                with st.spinner("Executing restoration & neural colorization pipeline…"):
                    net = get_cached_model() if colorize_enabled else None
                    result, elapsed_seconds = run_pipeline(
                        image_np, net,
                        denoise=denoise_flag, contrast=contrast_flag, sharpen=sharpen_flag,
                        colorize_enabled=colorize_enabled
                    )
            except FileNotFoundError as e:
                st.error(f"Model file error: {e}")
            except Exception as e:
                st.error(f"Pipeline error: {e}")
            else:
                with st.container(border=True):
                    result_title = "🌈 Restored & Colorized Output" if colorize_enabled else "🛠️ Restored Output"
                    st.markdown(f"#### {result_title}")

                    render_before_after(image_np, result, "Original Photo", "Enhanced Output")

                    st.divider()

                    sharpness_after = estimate_sharpness(result)
                    contrast_after = estimate_contrast(result)
                    sharpness_delta = sharpness_after - estimate_sharpness(image_np)
                    contrast_delta = contrast_after - estimate_contrast(image_np)

                    metric_cols = st.columns(5)
                    metric_cols[0].metric("Dimensions", f"{result.shape[1]} × {result.shape[0]}")
                    metric_cols[1].metric("Processing Latency", f"{elapsed_seconds:.2f} s")
                    metric_cols[2].metric("Sharpness*", f"{sharpness_after:.1f}", delta=f"{sharpness_delta:+.1f}")
                    metric_cols[3].metric("Contrast*", f"{contrast_after:.1f}", delta=f"{contrast_delta:+.1f}")
                    metric_cols[4].metric("Device", get_processing_device())

                    st.caption(
                        "*Heuristic CV metrics: Laplacian variance for edge definition, pixel standard "
                        "deviation for global dynamic contrast."
                    )

                    download_name = "restored_colorized.png" if colorize_enabled else "restored.png"
                    st.download_button(
                        label="📥 Download High-Res Result",
                        data=encode_png(result),
                        file_name=download_name,
                        mime="image/png"
                    )

                    st.success("Enhancement pipeline completed successfully!")

# ===========================================================================
# 2. Batch Processing Studio
# ===========================================================================
with tab_batch:
    with st.container(border=True):
        st.markdown(f"#### 📤 Upload Batch Collection (Up to {MAX_BATCH_FILES} Images)")
        batch_files = st.file_uploader(
            f"Select up to {MAX_BATCH_FILES} images (JPG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="batch_uploader"
        )

    if batch_files:
        if len(batch_files) > MAX_BATCH_FILES:
            st.warning(
                f"Selected {len(batch_files)} images — only the first "
                f"{MAX_BATCH_FILES} will be processed to manage memory efficiently."
            )
            batch_files = batch_files[:MAX_BATCH_FILES]

        st.caption(f"{len(batch_files)} photo(s) queued for processing.")
        process_all = st.button("✨ Process Batch Collection", type="primary")

        if process_all:
            try:
                ensure_model_ready()
            except Exception:
                pass

            progress_label = st.empty()
            progress_bar = st.progress(0.0)
            net = get_cached_model() if colorize_enabled else None

            results = []
            for i, file in enumerate(batch_files):
                progress_label.text(f"Processing photo {i + 1} of {len(batch_files)}: {file.name}")
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

            progress_label.text(f"Batch completed: {len(batch_files)} / {len(batch_files)} processed.")

            succeeded = [(name, data) for name, data, err in results if err is None]
            failed = [(name, err) for name, data, err in results if err is not None]

            with st.container(border=True):
                st.markdown("#### 📦 Batch Processing Gallery")
                st.caption(f"Status: {len(succeeded)} succeeded, {len(failed)} failed.")

                if succeeded:
                    st.download_button(
                        "📥 Download All as ZIP Archive",
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
                            "Download PNG", data=data, file_name=f"restored_{name}",
                            mime="image/png", key=f"dl_{idx}_{name}"
                        )

                for name, err in failed:
                    st.error(f"**{name}**: {err}")
