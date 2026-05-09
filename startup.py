"""
Download model.safetensors from Google Drive if not already present.
Uses gdown which correctly handles Google Drive large-file confirmation pages.

Set this environment variable on your deployment platform:
  GDRIVE_FILE_ID  — the file ID from your Google Drive share link
                    e.g. for https://drive.google.com/file/d/ABC123/view
                    the ID is: ABC123
"""

import os
import sys

MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.safetensors")
GDRIVE_FILE_ID = os.environ.get("GDRIVE_FILE_ID", "")

# Minimum expected size for a valid model file (100 MB).
# If the downloaded file is smaller, Google Drive returned an HTML page instead.
MIN_MODEL_BYTES = 100 * 1024 * 1024


def download_model():
    # Skip if a valid model file is already present.
    if os.path.isfile(MODEL_FILE) and os.path.getsize(MODEL_FILE) > MIN_MODEL_BYTES:
        print(f"[startup] model.safetensors already present "
              f"({os.path.getsize(MODEL_FILE) // 1024 // 1024} MB), skipping download.")
        return

    # Remove a corrupt/partial file from a previous failed attempt.
    if os.path.isfile(MODEL_FILE):
        print("[startup] Found incomplete model file — removing and re-downloading.")
        os.remove(MODEL_FILE)

    if not GDRIVE_FILE_ID:
        print("[startup] GDRIVE_FILE_ID not set — skipping download (HF Hub fallback will be used).")
        return

    print(f"[startup] Downloading model.safetensors via gdown (ID={GDRIVE_FILE_ID}) ...")

    try:
        import gdown
    except ImportError:
        print("[startup] ERROR: gdown not installed. Add gdown>=4.7.3 to requirements.txt.")
        sys.exit(1)

    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    gdown.download(url, MODEL_FILE, quiet=False)

    # Validate the downloaded file is actually the model (not an HTML error page).
    if not os.path.isfile(MODEL_FILE) or os.path.getsize(MODEL_FILE) < MIN_MODEL_BYTES:
        size = os.path.getsize(MODEL_FILE) if os.path.isfile(MODEL_FILE) else 0
        print(f"[startup] ERROR: Downloaded file is only {size // 1024} KB — "
              f"Google Drive likely returned an HTML error page.\n"
              f"Make sure the file is shared as 'Anyone with the link' (Viewer).")
        if os.path.isfile(MODEL_FILE):
            os.remove(MODEL_FILE)
        sys.exit(1)

    print(f"[startup] Download complete: "
          f"{os.path.getsize(MODEL_FILE) // 1024 // 1024} MB saved to {MODEL_FILE}")


if __name__ == "__main__":
    download_model()
