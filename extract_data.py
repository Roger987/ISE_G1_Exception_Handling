from huggingface_hub import snapshot_download
import py7zr, os

snapshot_download(
    repo_id="harisec/vibe-coded-web-apps",
    local_dir="vibe_dataset",
    repo_type="dataset"
)

with py7zr.SevenZipFile("vibe_dataset/vibe-coded-apps.7z", mode='r') as z:
    z.extractall(path="vibe_dataset")


os.remove("vibe_dataset/vibe-coded-web-apps.7z")
print("Dataset extracted to vibe_dataset/")