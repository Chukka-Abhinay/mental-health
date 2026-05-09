"""
Download model.safetensors from Google Drive if not already present.
Called automatically at app startup via startup.py or directly.

Set these environment variables on your deployment platform:
  GDRIVE_FILE_ID  — the file ID from your Google Drive share link
                    e.g. for https://drive.google.com/file/d/ABC123/view
                    the ID is: ABC123
"""

import os
import sys

MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.safetensors")
GDRIVE_FILE_ID = os.environ.get("GDRIVE_FILE_ID", "")


def download_model():
    if os.path.isfile(MODEL_FILE):
        print(f"[startup] model.safetensors already present ({os.path.getsize(MODEL_FILE)//1024//1024} MB), skipping download.")
        return

    if not GDRIVE_FILE_ID:
        print("[startup] GDRIVE_FILE_ID not set — skipping model download (will try HF Hub fallback).")
        return

    print(f"[startup] Downloading model.safetensors from Google Drive (ID={GDRIVE_FILE_ID}) ...")

    try:
        import requests
    except ImportError:
        print("[startup] ERROR: requests not installed.")
        sys.exit(1)

    # Google Drive large-file download with confirmation token handling
    session = requests.Session()
    url = "https://drive.google.com/uc?export=download"

    response = session.get(url, params={"id": GDRIVE_FILE_ID}, stream=True)

    # Check for virus-scan warning page (large files)
    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break

    if token:
        response = session.get(url, params={"id": GDRIVE_FILE_ID, "confirm": token}, stream=True)

    # Stream to disk
    total = 0
    with open(MODEL_FILE, "wb") as f:
        for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):  # 8 MB chunks
            if chunk:
                f.write(chunk)
                total += len(chunk)
                print(f"\r[startup] Downloaded {total // 1024 // 1024} MB ...", end="", flush=True)

    print(f"\n[startup] Download complete: {total // 1024 // 1024} MB saved to {MODEL_FILE}")


if __name__ == "__main__":
    download_model()
