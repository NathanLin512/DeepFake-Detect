import os
import requests
import zipfile

# 定義目錄結構
base_dir = "HDDM_Project/data/Dataset"
real_dir = os.path.join(base_dir, "real")
fake_dir = os.path.join(base_dir, "fake")

os.makedirs(real_dir, exist_ok=True)
os.makedirs(fake_dir, exist_ok=True)

# 定義 Hugging Face 壓縮檔案的 URL
files = {
    "wiki.zip": real_dir,  # Real images
    "inpainting.zip": os.path.join(fake_dir, "inpainting"),
    "insight.zip": os.path.join(fake_dir, "insight"),
    "text2img.zip": os.path.join(fake_dir, "text2img"),
}

base_url = "https://huggingface.co/datasets/OpenRL/DeepFakeFace/resolve/main/"

# 下載並解壓縮
def download_and_extract(file_name, target_dir):
    url = base_url + file_name
    local_zip_path = os.path.join(base_dir, file_name)

    # 下載檔案
    if not os.path.exists(local_zip_path):
        print(f"Downloading {file_name}...")
        response = requests.get(url, stream=True)
        with open(local_zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded {file_name}.")

    # 解壓縮檔案
    print(f"Extracting {file_name}...")
    os.makedirs(target_dir, exist_ok=True)
    with zipfile.ZipFile(local_zip_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)
    print(f"Extracted {file_name} to {target_dir}.")

# 處理所有檔案
for file_name, target_dir in files.items():
    download_and_extract(file_name, target_dir)

print("All files downloaded and extracted successfully.")
