"""Downloads the colorization model weights on first run, if missing.

The weights (~123MB) aren't committed to the repo -- see the README's
'Model Setup' section for why. This module fetches them automatically so
a fresh clone works out of the box; if the download fails for any reason,
the caller falls back to the existing manual-download instructions rather
than crashing.
"""

import shutil
import urllib.error
import urllib.request

from .config import (
    MIN_MODEL_SIZE_BYTES,
    MODEL_DOWNLOAD_TIMEOUT_SECONDS,
    MODEL_FALLBACK_URL,
    MODEL_PATH,
    MODEL_URL,
)


def ensure_model_downloaded(
    progress_callback=None,
    url=MODEL_URL,
    destination=MODEL_PATH,
    min_size_bytes=MIN_MODEL_SIZE_BYTES,
    timeout=MODEL_DOWNLOAD_TIMEOUT_SECONDS,
):
    """Make sure the model weights exist at `destination`, downloading if not.

    Returns True if the weights are present after this call (already there,
    or just downloaded successfully), False if they're missing and the
    download failed. Never raises: network/IO failures are treated as "the
    weights aren't available" so the caller can fall back to manual
    instructions, rather than letting an exception crash the app.

    `progress_callback(bytes_downloaded, total_bytes)`, if given, is called
    after each chunk; `total_bytes` is 0 if the server didn't report a
    Content-Length.
    """
    if destination.is_file():
        if destination.stat().st_size >= min_size_bytes:
            return True
        destination.unlink(missing_ok=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".partial")

    urls = (url, MODEL_FALLBACK_URL) if url == MODEL_URL else (url,)
    for download_url in urls:
        try:
            tmp_path.unlink(missing_ok=True)
            request = urllib.request.Request(
                download_url, headers={"User-Agent": "PixelRevive-AI/1.0"}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                total = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                with open(tmp_path, "wb") as f:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback:
                            progress_callback(downloaded, total)

            if tmp_path.stat().st_size < min_size_bytes:
                continue

            shutil.move(str(tmp_path), str(destination))
            return True
        except (urllib.error.URLError, OSError, ValueError, TimeoutError):
            pass
        finally:
            tmp_path.unlink(missing_ok=True)

    return False
