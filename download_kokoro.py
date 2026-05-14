import os
import urllib.request
import logging

logging.basicConfig(level=logging.INFO)

MODELS_DIR = "kokoro_models"
os.makedirs(MODELS_DIR, exist_ok=True)

FILES = {
    "kokoro-v1.0.int8.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx",
    "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
}

def download_file(url, dest):
    if os.path.exists(dest):
        logging.info(f"File {dest} already exists. Skipping download.")
        return
    logging.info(f"Downloading {url} to {dest}...")
    urllib.request.urlretrieve(url, dest)
    logging.info(f"Downloaded {dest} successfully.")

if __name__ == "__main__":
    for filename, url in FILES.items():
        dest_path = os.path.join(MODELS_DIR, filename)
        try:
            download_file(url, dest_path)
        except Exception as e:
            logging.error(f"Failed to download {filename}: {e}")
            logging.error("If the GitHub release is down, please manually download the ONNX files to 'kokoro_models' folder.")
