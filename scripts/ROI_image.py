'''
Ref. https://github.com/PyImageSearch/imutils/tree/master
'''
import os
import cv2
import dlib
from imutils import face_utils

# 定義路徑
base_dir = "HDDM_Project/data/processed"
real_dir = os.path.join(base_dir, "real")
fake_dir = os.path.join(base_dir, "fake")

output_dir = "HDDM_Project/data/ROI_data"
output_real_dir = os.path.join(output_dir, "real")
output_fake_dir = os.path.join(output_dir, "fake")

os.makedirs(output_real_dir, exist_ok=True)
os.makedirs(output_fake_dir, exist_ok=True)

# 初始化 Dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 定義函式
def extract_roi(image, landmarks, points):
    roi_points = landmarks[points]
    x, y, w, h = cv2.boundingRect(roi_points)
    return image[y:y + h, x:x + w]

def process_images(input_dir, output_dir):
    error_files = []
    for file_name in os.listdir(input_dir):
        if not file_name.endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(input_dir, file_name)
        image = cv2.imread(image_path)

        if image is None:
            error_files.append(file_name)
            print(f"Warning: Unable to read image {file_name}. Skipping...")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)

            try:
                left_eye = extract_roi(image, landmarks, range(36, 42))
                right_eye = extract_roi(image, landmarks, range(42, 48))
                nose = extract_roi(image, landmarks, range(27, 36))
                mouth = extract_roi(image, landmarks, range(48, 68))

                cv2.imwrite(os.path.join(output_dir, f"left_eye_{file_name}"), left_eye)
                cv2.imwrite(os.path.join(output_dir, f"right_eye_{file_name}"), right_eye)
                cv2.imwrite(os.path.join(output_dir, f"nose_{file_name}"), nose)
                cv2.imwrite(os.path.join(output_dir, f"mouth_{file_name}"), mouth)
            except Exception as e:
                print(f"Error processing ROI for {file_name}: {e}")

    # 儲存損壞檔案
    if error_files:
        with open("error_files.log", "w") as f:
            f.write("\n".join(error_files))
        print(f"Error log saved with {len(error_files)} corrupted files.")

# 處理 real 和 fake 資料
process_images(real_dir, output_real_dir)
process_images(fake_dir, output_fake_dir)
