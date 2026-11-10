import io
from pathlib import Path

import numpy as np
import cv2
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# streamlit-drawable-canvas 0.9.3 calls a Streamlit image helper that was
# removed in Streamlit 1.62. Keep its existing canvas API working by adapting
# that call to Streamlit's replacement helper.
import streamlit.elements.image as st_image

if not hasattr(st_image, "image_to_url"):
    from streamlit.elements.lib.image_utils import image_to_url
    from streamlit.elements.lib.layout_utils import LayoutConfig

    def _canvas_image_to_url(image, width, clamp, channels, output_format, image_id):
        return image_to_url(
            image,
            LayoutConfig(width=width),
            clamp,
            channels,
            output_format,
            image_id,
        )

    st_image.image_to_url = _canvas_image_to_url

from src.inpainting import remove_object
from src.background_removal import remove_background
from src.edge_detection import detect_edges
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

SAMPLE_IMAGES_DIR = Path(__file__).resolve().parent / "test_images_features"

st.set_page_config(
    page_title="PixelRevive AI",
    layout="wide"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #f4f6fb; }
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1320px; padding-top: 1.8rem; padding-bottom: 3rem; }
h1, h2, h3 { color: #202d47; letter-spacing: -0.025em; }
h1 { font-weight: 760; }
.brand-lockup {
    display: flex; align-items: center; gap: 1.1rem; margin: .2rem 0 1.2rem;
    padding: 1.15rem 1.45rem; border: 1px solid #e4e9f2; border-radius: 20px;
    background: linear-gradient(115deg, #fff 0%, #f7f8ff 58%, #f1f5ff 100%);
    box-shadow: 0 12px 30px rgba(30, 48, 78, .06);
}
.brand-sticker {
    width: 62px; height: 62px; flex: 0 0 62px; display: grid; place-items: center;
    border-radius: 18px; background: linear-gradient(140deg, #ffead8, #e8e7ff);
    box-shadow: inset 0 0 0 1px rgba(255,255,255,.9); font-size: 1.9rem;
}
.section-intro {
    display: flex; align-items: center; gap: .9rem; margin: .35rem 0 1rem;
    padding: .9rem 1.1rem; background: #fff; border: 1px solid #e4e9f2;
    border-radius: 15px; box-shadow: 0 5px 18px rgba(30,48,78,.035);
}
.section-icon {
    width: 46px; height: 46px; flex: 0 0 46px; display: grid; place-items: center;
    border-radius: 14px; background: #eef1ff; font-size: 1.45rem;
}
.section-title { color: #202d47; font-size: 1.16rem; font-weight: 700; }
.section-description { margin-top: .12rem; color: #73809a; font-size: .9rem; }
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #fff; border: 1px solid #e3e8f1; border-radius: 16px;
    box-shadow: 0 7px 22px rgba(30, 48, 78, .04);
}
[data-testid="stSidebar"] { background: #eef1f7; border-right: 1px solid #e2e7f0; }
[data-testid="stFileUploader"] section {
    background: #f8faff; border: 1px dashed #cbd5e5; border-radius: 13px;
}
div.stButton > button, div.stDownloadButton > button {
    min-height: 2.7rem; border-radius: 11px; font-weight: 650;
    transition: transform 120ms ease, box-shadow 120ms ease, background 120ms ease;
}
[data-testid="stBaseButton-primary"] {
    background: #30456b !important; border-color: #30456b !important; color: #fff !important;
}
[data-testid="stBaseButton-primary"]:hover {
    background: #263958 !important; border-color: #263958 !important;
}
div.stButton > button:hover, div.stDownloadButton > button:hover {
    transform: translateY(-1px); box-shadow: 0 5px 14px rgba(39, 59, 91, .12);
}
[data-testid="stMetric"] {
    padding: .8rem; background: #f8faff; border: 1px solid #e8edf5;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="brand-lockup">
  <div class="brand-sticker">✨</div>
  <div>
    <div style="color:#60718e; font-size:.76rem; font-weight:700; letter-spacing:.15em; text-transform:uppercase;">Image restoration studio</div>
    <h1 style="margin:.18rem 0 .2rem;">PixelRevive AI</h1>
    <div style="color:#65738a; font-size:1.04rem;">AI-Powered Image Restoration, Enhancement &amp; Processing Studio</div>
  </div>
</div>
""", unsafe_allow_html=True)

sections = [
    "Single Image",
    "Batch Processing",
    "Object & Scratch Removal",
    "Background Removal",
    "Edge Detection",
]
section_icons = ["🖼️", "🗂️", "🪄", "✂️", "🔎"]
section_descriptions = [
    "Restore, enhance, and colorize one photo.",
    "Apply the same enhancement workflow to multiple photos.",
    "Paint over unwanted objects or scratches to remove them.",
    "Select a subject and create a transparent background.",
    "Find outlines and structures with configurable edge tools.",
]
selected_section = st.session_state.get("main_section", sections[0])
nav_columns = st.columns(len(sections), gap="small")
for index, (column, section) in enumerate(zip(nav_columns, sections)):
    with column:
        if st.button(
            f"{section_icons[index]}  {section}",
            key=f"nav_{index}",
            type="primary" if selected_section == section else "secondary",
            use_container_width=True,
        ):
            st.session_state["main_section"] = section
            st.rerun()

selected_section = st.session_state.get("main_section", sections[0])
selected_index = sections.index(selected_section)
st.markdown(f"""
<div class="section-intro">
  <div class="section-icon">{section_icons[selected_index]}</div>
  <div>
    <div class="section-title">{selected_section}</div>
    <div class="section-description">{section_descriptions[selected_index]}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# These controls apply to the enhancement and colorization workflows only.
if selected_section in sections[:2]:
    st.sidebar.header("Processing")
    st.sidebar.caption("Choose which steps to apply before colorizing.")

    restoration_enabled = st.sidebar.checkbox("Restoration", value=True)
    denoise_enabled = st.sidebar.checkbox(
        "Denoising", value=True, disabled=not restoration_enabled
    )
    contrast_enabled = st.sidebar.checkbox(
        "Contrast Enhancement", value=True, disabled=not restoration_enabled
    )
    sharpen_enabled = st.sidebar.checkbox(
        "Sharpening", value=True, disabled=not restoration_enabled
    )

    st.sidebar.divider()
    colorize_enabled = st.sidebar.checkbox("AI Colorization", value=True)

    denoise_flag = denoise_enabled and restoration_enabled
    contrast_flag = contrast_enabled and restoration_enabled
    sharpen_flag = sharpen_enabled and restoration_enabled
else:
    restoration_enabled = False
    colorize_enabled = False
    denoise_flag = contrast_flag = sharpen_flag = False


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


def sample_image_upload(folder_name, key):
    """Offer a sample from the matching feature folder as an upload-like file."""
    sample_dir = SAMPLE_IMAGES_DIR / folder_name
    sample_paths = sorted(
        path for path in sample_dir.glob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ) if sample_dir.is_dir() else []
    if not sample_paths:
        return None

    selected_path = st.selectbox(
        "Or try a sample image",
        [None, *sample_paths],
        format_func=lambda path: "Choose a sample..." if path is None else path.stem,
        key=key,
    )
    if selected_path is None:
        return None

    sample_file = io.BytesIO(selected_path.read_bytes())
    sample_file.name = selected_path.name
    sample_file.size = len(sample_file.getbuffer())
    return sample_file


# ===========================================================================
# Single image
# ===========================================================================
if selected_section == sections[0]:

    with st.container(border=True):
        st.markdown("### 🖼️ Upload a Photo")
        sample_file = sample_image_upload("processing-colorization", "single_sample")
        uploaded_file = st.file_uploader(
            "Drag and drop or browse — JPG, PNG, or WEBP",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
            key="single_uploader"
        )
        uploaded = uploaded_file if uploaded_file is not None else sample_file

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
            st.markdown("### 🪄 Preview")
            st.image(image_np, use_container_width=True)
            run = st.button("Restore & Colorize", type="primary")

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

                    result_heading = "Restored & Colorized" if colorize_enabled else "Restored"
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
                        label="Download Result",
                        data=encode_png(result),
                        file_name=download_name,
                        mime="image/png"
                    )

                    st.success("Done!")

# ===========================================================================
# Batch processing
# ===========================================================================
if selected_section == sections[1]:

    with st.container(border=True):
        st.markdown("### 🗂️ Upload Multiple Photos")
        uploaded_batch = st.file_uploader(
            f"Up to {MAX_BATCH_FILES} images — JPG, PNG, or WEBP",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="batch_uploader"
        )
        batch_files = uploaded_batch

    if batch_files:

        if len(batch_files) > MAX_BATCH_FILES:
            st.warning(
                f"{len(batch_files)} images selected — only the first "
                f"{MAX_BATCH_FILES} will be processed to keep memory and "
                f"processing time bounded."
            )
            batch_files = batch_files[:MAX_BATCH_FILES]

        st.caption(f"{len(batch_files)} image(s) ready.")
        process_all = st.button("Process All", type="primary")

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
                        "Download All as ZIP",
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


# ===========================================================================
# Object and Scratch Removal
# ===========================================================================

# ===========================================================================
# Object and Scratch Removal
# ===========================================================================

if selected_section == sections[2]:

    with st.container(border=True):
        st.markdown("### Choose a photo")
        st.write("Draw over objects or scratches you want to remove.")
        sample_file = sample_image_upload("scratch-object-removal", "inpaint_sample")
        uploaded_inpaint_file = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png", "webp"],
            key="inpaint_uploader"
        )
        uploaded_inpaint = uploaded_inpaint_file if uploaded_inpaint_file is not None else sample_file

    if uploaded_inpaint is not None:

        try:
            # Read image
            image_np, _ = read_upload_as_rgb(uploaded_inpaint)
            image_np, was_resized = resize_if_too_large(image_np)

            original_height, original_width = image_np.shape[:2]

            # Prepare image for canvas
            display_image = Image.fromarray(image_np)
            display_image.thumbnail((800, 800))

            canvas_width, canvas_height = display_image.size

            st.caption(
                f"Image Resolution: {original_width} × {original_height}"
            )

            if was_resized:
                st.info("Image was resized for processing.")

            st.markdown("#### 1. Select the region to remove")

            st.info(
                "Draw over the unwanted object or scratches. "
                "Selected regions will appear red."
            )

            # Brush size
            brush_size = st.slider(
                "Brush Size",
                min_value=3,
                max_value=50,
                value=15,
                key="inpaint_brush"
            )

            # Drawing canvas
            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.0)",
                stroke_width=brush_size,
                stroke_color="#FF0000",
                background_image=display_image,
                background_color="#FFFFFF",
                height=canvas_height,
                width=canvas_width,
                drawing_mode="freedraw",
                key="inpaint_canvas"
            )

            st.markdown("#### 2. Restoration Settings")

            col1, col2 = st.columns(2)

            with col1:
                method = st.selectbox(
                    "Inpainting Algorithm",
                    ["Telea", "Navier-Stokes"],
                    key="inpaint_method"
                )

            with col2:
                radius = st.slider(
                    "Inpainting Radius",
                    min_value=1,
                    max_value=15,
                    value=5,
                    key="inpaint_radius"
                )

            # Restore button
            if st.button(
                "Remove Selected Region",
                type="primary",
                key="remove_object_button"
            ):

                if canvas_result.image_data is None:
                    st.warning("Please draw over the image first.")

                else:
                    canvas_data = canvas_result.image_data

                    # Extract RGB channels
                    canvas_rgb = canvas_data[:, :, :3].astype(np.int16)

                    # Detect red brush strokes
                    red_mask = (
                        (canvas_rgb[:, :, 0] > 150) &
                        (canvas_rgb[:, :, 1] < 120) &
                        (canvas_rgb[:, :, 2] < 120)
                    )

                    mask_small = (
                        red_mask.astype(np.uint8) * 255
                    )

                    if not np.any(mask_small):
                        st.warning(
                            "No selection detected. Please draw "
                            "over the region using the red brush."
                        )

                    else:
                        # Resize mask to original image dimensions
                        mask = cv2.resize(
                            mask_small,
                            (original_width, original_height),
                            interpolation=cv2.INTER_NEAREST
                        )

                        # Slightly expand mask to cover brush edges
                        kernel = np.ones((3, 3), np.uint8)

                        mask = cv2.dilate(
                            mask,
                            kernel,
                            iterations=1
                        )

                        # Display mask for verification
                        with st.expander("View selected mask"):
                            st.image(
                                mask,
                                caption="White = Selected Region",
                                clamp=True
                            )

                        # Perform inpainting
                        with st.spinner(
                            "Removing selected region..."
                        ):

                            restored = remove_object(
                                image_np,
                                mask,
                                radius=radius,
                                method=method
                            )

                        st.success("Restoration completed!")

                        st.markdown("### 3. Restoration Result")

                        # Before and after comparison
                        render_before_after(
                            image_np,
                            restored,
                            "Original",
                            "Restored"
                        )

                        st.divider()

                        st.image(
                            restored,
                            caption="Restored Image",
                            use_container_width=True
                        )

                        # Download restored image
                        st.download_button(
                            label="Download Restored Image",
                            data=encode_png(restored),
                            file_name="object_removed.png",
                            mime="image/png",
                            key="download_inpaint"
                        )

        except Exception as e:
            st.error(f"Error during object removal: {e}")




# ===========================================================================
# Background Removal
# ===========================================================================

# ===========================================================================
# Background Removal with Mouse-Based Rectangle Selection
# ===========================================================================

if selected_section == sections[3]:

    with st.container(border=True):
        st.markdown("### Choose a photo")
        st.write("Mark a rectangle around the subject you want to keep.")
        sample_file = sample_image_upload("background-removal", "background_sample")
        uploaded_background_file = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png", "webp"],
            key="background_uploader"
        )
        uploaded_background = uploaded_background_file if uploaded_background_file is not None else sample_file

    if uploaded_background is not None:

        try:
            # Read image
            image_np, _ = read_upload_as_rgb(uploaded_background)

            image_np, _ = resize_if_too_large(image_np)

            height, width = image_np.shape[:2]

            # Prepare image for drawing canvas
            display_image = Image.fromarray(image_np)
            display_image.thumbnail((800, 800))

            canvas_width, canvas_height = display_image.size

            st.markdown("#### 1. Select the foreground subject")

            st.info(
                "Use your mouse to draw a rectangle around the "
                "complete object you want to keep."
            )

            # Rectangle drawing canvas
            canvas_result = st_canvas(
                fill_color="rgba(0, 255, 0, 0.10)",
                stroke_width=3,
                stroke_color="#00FF00",
                background_image=display_image,
                background_color="#FFFFFF",
                height=canvas_height,
                width=canvas_width,
                drawing_mode="rect",
                update_streamlit=True,
                key="background_rectangle_canvas"
            )

            st.caption(
                f"Original image dimensions: {width} × {height}"
            )

            iterations = st.slider(
                "Processing Iterations",
                min_value=1,
                max_value=10,
                value=5,
                key="background_iterations"
            )

            if st.button(
                "Remove Background",
                type="primary",
                key="background_remove_button"
            ):

                rect = None

                # ---------------------------------------------------
                # Method 1: Extract rectangle from canvas JSON
                # ---------------------------------------------------

                if canvas_result.json_data is not None:

                    objects = canvas_result.json_data.get(
                        "objects", []
                    )

                    rectangles = [
                        obj for obj in objects
                        if obj.get("type") == "rect"
                    ]

                    if rectangles:

                        selected = rectangles[-1]

                        left = float(selected.get("left", 0))
                        top = float(selected.get("top", 0))

                        rect_width = (
                            float(selected.get("width", 0))
                            * float(selected.get("scaleX", 1))
                        )

                        rect_height = (
                            float(selected.get("height", 0))
                            * float(selected.get("scaleY", 1))
                        )

                        scale_x = width / canvas_width
                        scale_y = height / canvas_height

                        x = round(left * scale_x)
                        y = round(top * scale_y)

                        w = round(rect_width * scale_x)
                        h = round(rect_height * scale_y)

                        rect = (x, y, w, h)

                # ---------------------------------------------------
                # Method 2: Fallback - detect green rectangle pixels
                # ---------------------------------------------------

                if rect is None and canvas_result.image_data is not None:

                    canvas_rgb = canvas_result.image_data[:, :, :3]

                    green_pixels = (
                        (canvas_rgb[:, :, 0] < 100) &
                        (canvas_rgb[:, :, 1] > 180) &
                        (canvas_rgb[:, :, 2] < 100)
                    )

                    # Find connected green components
                    green_mask = green_pixels.astype(np.uint8)

                    count, labels, stats, centroids = (
                        cv2.connectedComponentsWithStats(
                            green_mask,
                            connectivity=8
                        )
                    )

                    if count > 1:

                        # Select largest connected component
                        largest = 1 + np.argmax(
                            stats[1:, cv2.CC_STAT_AREA]
                        )

                        if stats[largest, cv2.CC_STAT_AREA] >= 20:

                            x1 = stats[largest, cv2.CC_STAT_LEFT]
                            y1 = stats[largest, cv2.CC_STAT_TOP]

                            x2 = x1 + stats[
                                largest, cv2.CC_STAT_WIDTH
                            ]

                            y2 = y1 + stats[
                                largest, cv2.CC_STAT_HEIGHT
                            ]

                            scale_x = width / canvas_width
                            scale_y = height / canvas_height

                            x = round(x1 * scale_x)
                            y = round(y1 * scale_y)

                            w = round((x2 - x1) * scale_x)
                            h = round((y2 - y1) * scale_y)

                            rect = (x, y, w, h)

                # ---------------------------------------------------
                # Validate and process selection
                # ---------------------------------------------------

                if rect is None:

                    st.warning(
                        "Rectangle not detected. Please draw a "
                        "rectangle around the subject again."
                    )

                else:

                    x, y, w, h = rect

                    x = max(0, min(x, width - 1))
                    y = max(0, min(y, height - 1))

                    w = min(w, width - x)
                    h = min(h, height - y)

                    if w < 2 or h < 2:

                        st.error(
                            "Invalid rectangle. Please draw a larger rectangle."
                        )

                    else:

                        st.write(
                            f"Selected region: X={x}, Y={y}, "
                            f"Width={w}, Height={h}"
                        )

                        with st.spinner("Removing background..."):

                            result = remove_background(
                                image_np,
                                (x, y, w, h),
                                iterations=int(iterations)
                            )

                        st.session_state["background_result"] = result
                        st.session_state["background_source"] = uploaded_background.name

                        st.success("Background removed successfully!")

            # ---------------------------------------------------
            # Display result
            # ---------------------------------------------------

            result = st.session_state.get("background_result")

            result_source = st.session_state.get("background_source")

            if (
                result is not None
                and result_source == uploaded_background.name
            ):

                st.markdown("### 2. Transparent Background Result")

                st.image(
                    result,
                    caption="Background Removed",
                    use_container_width=True
                )

                st.markdown("### 3. Replace Background")

                preview_color = st.color_picker(
                    "Choose a new background color",
                    "#FFFFFF",
                    key="background_color"
                )

                color_rgb = tuple(
                    int(preview_color[i:i + 2], 16)
                    for i in (1, 3, 5)
                )

                alpha = (
                    result[:, :, 3:4].astype(np.float32) / 255.0
                )

                new_background = np.full(
                    result[:, :, :3].shape,
                    color_rgb,
                    dtype=np.uint8
                )

                preview = (
                    result[:, :, :3].astype(np.float32) * alpha
                    + new_background.astype(np.float32) * (1 - alpha)
                ).clip(0, 255).astype(np.uint8)

                st.image(
                    preview,
                    caption="Image with New Background",
                    use_container_width=True
                )

                st.download_button(
                    label="Download Transparent PNG",
                    data=encode_png(result),
                    file_name="background_removed.png",
                    mime="image/png",
                    key="download_background"
                )

        except Exception as e:

            st.error(f"Background removal error: {e}")






# ===========================================================================
# Edge Detection Studio
# ===========================================================================

if selected_section == sections[4]:

    with st.container(border=True):
        st.markdown("### Choose a photo")
        st.write("Detect boundaries and important structures in your photo.")
        sample_file = sample_image_upload("edge-detection", "edge_sample")
        uploaded_edge_file = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png", "webp"],
            key="edge_uploader"
        )
        uploaded_edge = uploaded_edge_file if uploaded_edge_file is not None else sample_file

    if uploaded_edge is not None:

        try:
            # Read image
            image_np, _ = read_upload_as_rgb(uploaded_edge)

            image_np, _ = resize_if_too_large(image_np)

            st.markdown("#### 1. Original Image")

            st.image(
                image_np,
                caption="Uploaded Image",
                use_container_width=True
            )

            st.markdown("#### 2. Detection Settings")

            method = st.selectbox(
                "Select Edge Detection Algorithm",
                ["Sobel", "Prewitt", "Canny", "Laplacian"],
                key="edge_method"
            )

            threshold1 = 100
            threshold2 = 200
            kernel_size = 3

            if method == "Canny":

                st.markdown("##### Canny Thresholds")

                col1, col2 = st.columns(2)

                with col1:
                    threshold1 = st.slider(
                        "Lower Threshold",
                        min_value=0,
                        max_value=255,
                        value=100,
                        key="canny_threshold1"
                    )

                with col2:
                    threshold2 = st.slider(
                        "Upper Threshold",
                        min_value=0,
                        max_value=255,
                        value=200,
                        key="canny_threshold2"
                    )

                if threshold1 >= threshold2:
                    st.warning(
                        "Upper threshold should be greater than lower threshold."
                    )

            if method in ["Sobel", "Laplacian"]:

                kernel_size = st.select_slider(
                    "Kernel Size",
                    options=[1, 3, 5, 7],
                    value=3,
                    key="edge_kernel"
                )

            if st.button(
                "Detect Edges",
                type="primary",
                key="detect_edges_button"
            ):

                if (
                    method == "Canny"
                    and threshold1 >= threshold2
                ):
                    st.error(
                        "Upper threshold must be greater than lower threshold."
                    )

                else:

                    with st.spinner("Detecting edges..."):

                        edges = detect_edges(
                            image_np,
                            method=method,
                            threshold1=threshold1,
                            threshold2=threshold2,
                            kernel_size=kernel_size
                        )

                    # Convert grayscale result to RGB
                    edges_rgb = cv2.cvtColor(
                        edges,
                        cv2.COLOR_GRAY2RGB
                    )

                    st.session_state["edge_result"] = edges_rgb
                    st.session_state["edge_source"] = uploaded_edge.name

                    st.success(
                        f"{method} edge detection completed!"
                    )

            # Display result
            result = st.session_state.get("edge_result")

            result_source = st.session_state.get("edge_source")

            if (
                result is not None
                and result_source == uploaded_edge.name
            ):

                st.markdown("### 3. Edge Detection Result")

                col1, col2 = st.columns(2)

                with col1:
                    st.image(
                        image_np,
                        caption="Original Image",
                        use_container_width=True
                    )

                with col2:
                    st.image(
                        result,
                        caption="Detected Edges",
                        use_container_width=True
                    )

                st.download_button(
                    label="Download Edge Image",
                    data=encode_png(result),
                    file_name="edge_detection.png",
                    mime="image/png",
                    key="download_edges"
                )

        except Exception as e:

            st.error(f"Edge detection error: {e}")
