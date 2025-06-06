import requests
import os
from time import sleep

repo_id = "vishalmlandge/vishal-models"
files = [
    ("best_model.pt", "."),
    ("Marathi_hindi_best_model.pt", "."),
    ("distilbert_local/model.safetensors", "distilbert_local"),
    ("improved_toxic_detector_model/model.safetensors", "improved_toxic_detector_model"),
    ("marathi_hindi_toxic_detector_model/pytorch_model.bin", "marathi_hindi_toxic_detector_model"),
    ("train.csv", ".")
]

def download_file(url, local_path, retries=3, timeout=60):
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"Downloaded {local_path}")
            return
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {url}: {e}. Retrying ({attempt+1}/{retries})...")
            sleep(5)  # Wait 5 seconds before retrying
    raise Exception(f"Failed to download {url} after {retries} retries")

for file, path in files:
    os.makedirs(path, exist_ok=True)
    file_url = f"https://huggingface.co/{repo_id}/resolve/main/{file}"
    local_path = os.path.join(path, os.path.basename(file))
    download_file(file_url, local_path)
