import os
import random
from PIL import Image

# 定義資料夾路徑
base_dir = "HDDM_Project/data/Dataset"
real_dir = os.path.join(base_dir, "real/wiki")
fake_dirs = [
    os.path.join(base_dir, "fake/inpainting"),
    os.path.join(base_dir, "fake/insight/insight"),
    os.path.join(base_dir, "fake/text2img/text2img"),
]
output_dir = "HDDM_Project/data/processed"

# 輸出資料夾
output_real_dir = os.path.join(output_dir, "real")
output_fake_dir = os.path.join(output_dir, "fake")
os.makedirs(output_real_dir, exist_ok=True)
os.makedirs(output_fake_dir, exist_ok=True)

# 定義影像resize大小
resize_size = (128, 128)

# Helper function: 收集資料夾中的所有影像檔案路徑
def collect_images_from_subfolders(root_dir):
    all_files = []
    for folder, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):
                all_files.append(os.path.join(folder, file))
    return all_files

# Helper function: Resize並儲存影像
def process_and_save_images(image_paths, output_dir, prefix, max_images, resize_size):
    count = 0
    for img_path in image_paths[:max_images]:
        try:
            img = Image.open(img_path).convert("RGB")
            img_resized = img.resize(resize_size)
            output_path = os.path.join(output_dir, f"{prefix}_{count}.jpg")
            img_resized.save(output_path)
            count += 1
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    print(f"Saved {count} images to {output_dir}")

# Step 1: 收集 real 和 fake 資料
print("Collecting real images...")
real_images = collect_images_from_subfolders(real_dir)
print(f"Found {len(real_images)} real images.")

print("Collecting fake images...")
fake_images = []
for fake_dir in fake_dirs:
    fake_images.extend(collect_images_from_subfolders(fake_dir))
print(f"Found {len(fake_images)} fake images.")

# Step 2: 確定要處理的影像數量
max_images = min(len(real_images), len(fake_images))  # 確保 real 和 fake 數量一致
print(f"Processing {max_images} images for each category.")

# Step 3: 隨機選擇影像
random.shuffle(real_images)
random.shuffle(fake_images)

# Step 4: Resize並儲存影像
print("Processing and saving real images...")
process_and_save_images(real_images, output_real_dir, prefix="real", max_images=max_images, resize_size=resize_size)

print("Processing and saving fake images...")
process_and_save_images(fake_images, output_fake_dir, prefix="fake", max_images=max_images, resize_size=resize_size)

print("Processing completed.")
