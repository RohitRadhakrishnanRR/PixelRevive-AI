# PixelRevive AI

**AI-Powered Image Restoration, Enhancement & Processing Studio**

PixelRevive AI is a Streamlit application for restoring, enhancing, colorizing, and processing photos. It combines OpenCV image processing with a pretrained neural network for grayscale colorization.

## Features

- **Single image processing:** apply denoising, contrast enhancement, sharpening, and AI colorization in a configurable workflow. Compare the original and result and download the output.
- **Batch processing:** process up to 10 images sequentially with progress, per-image results, and a ZIP download.
- **Object and scratch removal:** paint over unwanted areas and restore them with OpenCV inpainting (Telea or Navier–Stokes).
- **Background removal:** mark the foreground subject with a rectangle and create a transparent-background result using GrabCut.
- **Edge detection:** inspect image edges with configurable detection settings.
- **Image handling:** JPG, PNG, and WEBP uploads, image validation, and automatic resizing of large images.
- **Processing metrics:** view processing time, image dimensions, device information, and estimated sharpness and contrast.
- **Sample gallery:** choose bundled sample images from each feature's matching folder to try the tools without searching for a file.

The processing sidebar appears for **Single Image** and **Batch Processing**, where its enhancement and colorization controls apply.

## Requirements

- Python 3.12 or newer
- Dependencies listed in `requirements.txt`
- Internet access on the first colorization run to download the pretrained model (unless installed manually)

## Run locally

From the project directory:

```bash
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install dependencies and launch the app:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Colorization model

Colorization uses the Zhang, Isola, and Efros colorization network through OpenCV's DNN module. On first use, PixelRevive AI downloads the approximately 123 MB model file into `model/`. The network definition and color-bin data are included in the repository. If the automatic download is unavailable, download `colorization_release_v2.caffemodel` and place it in `model/`.

The model predicts color channels from image lightness, so its colors are plausible estimates rather than recovery of the photograph's original colors.

## Project layout

```text
app.py                  Streamlit interface and workflow orchestration
src/
  background_removal.py Background removal
  colorizer.py          Neural colorization
  edge_detection.py     Edge detection
  enhancer.py           Denoising, contrast, and sharpening
  image_utils.py        Image conversion, resizing, comparison, and ZIP helpers
  inpainting.py         Object and scratch removal
  metrics.py            Image quality estimates and device information
  model_downloader.py   Colorization model download
  pipeline.py           Shared enhancement and colorization pipeline
  config.py             Model paths and processing limits
model/                  Network definition and color-bin data
test_images_features/   Sample images grouped by feature
tests/                  Automated tests
```

## Tests

Run the automated test suite with:

```bash
pytest
```
