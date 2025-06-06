from huggingface_hub import hf_hub_download
import os

repo_id = "vishalmlandge/vishal-models"
files = [
    ("best_model.pt", "."),
    ("Marathi_hindi_best_model.pt", "."),
    ("distilbert_local/model.safetensors", "distilbert_local"),
    ("improved_toxic_detector_model/model.safetensors", "improved_toxic_detector_model"),
    ("marathi_hindi_toxic_detector_model/pytorch_model.bin", "marathi_hindi_toxic_detector_model"),
    ("train.csv", ".")
]
for file, path in files:
    os.makedirs(path, exist_ok=True)
    hf_hub_download(repo_id=repo_id, filename=file, local_dir=path)