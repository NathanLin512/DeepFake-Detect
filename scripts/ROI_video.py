import os
import cv2
import dlib
from imutils import face_utils

class VideoROIProcessor:
    def __init__(self, input_dir, output_dir, frame_interval=5, num_frames=16):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.frame_interval = frame_interval
        self.num_frames = num_frames

        # 初始化 dlib
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        # 創建目錄
        self.roi_categories = ["full_face", "left_eye", "right_eye", "nose", "mouth"]
        for category in ["real", "fake"]:
            for roi in self.roi_categories:
                os.makedirs(os.path.join(self.output_dir, category, roi), exist_ok=True)

    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video file: {video_path}")
            return []

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(frame_count // self.num_frames, self.frame_interval)

        frames = []
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % interval == 0:
                frames.append(frame)
                if len(frames) >= self.num_frames:
                    break

            frame_idx += 1
        cap.release()
        print(f"Extracted {len(frames)} frames from {video_path}")
        return frames

    def extract_roi(self, image, landmarks, points):
        roi_points = landmarks[points]
        x, y, w, h = cv2.boundingRect(roi_points)
        roi = image[y:y+h, x:x+w]
        return roi

    def process_single_video(self, video_path, category):
        print(f"Processing video: {video_path}")

        # 提取幀
        frames = self.extract_frames(video_path)
        for frame_idx, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)

            if len(faces) == 0:
                print(f"No faces detected in frame {frame_idx} of {os.path.basename(video_path)}")
                continue

            for face in faces:
                landmarks = self.predictor(gray, face)
                landmarks = face_utils.shape_to_np(landmarks)

                # 保存全臉
                full_face_path = os.path.join(self.output_dir, category, "full_face", f"{os.path.basename(video_path)}_frame_{frame_idx}.jpg")
                cv2.imwrite(full_face_path, frame)

                # 提取並保存 ROI
                try:
                    left_eye = self.extract_roi(frame, landmarks, range(36, 42))
                    right_eye = self.extract_roi(frame, landmarks, range(42, 48))
                    nose = self.extract_roi(frame, landmarks, range(27, 36))
                    mouth = self.extract_roi(frame, landmarks, range(48, 68))

                    cv2.imwrite(os.path.join(self.output_dir, category, "left_eye", f"{os.path.basename(video_path)}_frame_{frame_idx}.jpg"), left_eye)
                    cv2.imwrite(os.path.join(self.output_dir, category, "right_eye", f"{os.path.basename(video_path)}_frame_{frame_idx}.jpg"), right_eye)
                    cv2.imwrite(os.path.join(self.output_dir, category, "nose", f"{os.path.basename(video_path)}_frame_{frame_idx}.jpg"), nose)
                    cv2.imwrite(os.path.join(self.output_dir, category, "mouth", f"{os.path.basename(video_path)}_frame_{frame_idx}.jpg"), mouth)
                except Exception as e:
                    print(f"Error processing frame {frame_idx} in {os.path.basename(video_path)}: {e}")

    def process_videos(self):
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        for category in ["real", "fake"]:
            category_dir = os.path.join(self.input_dir, category)
            print(f"Processing category directory: {category_dir}")

            if not os.path.exists(category_dir):
                print(f"Category directory does not exist: {category_dir}")
                continue

            # 檢查是否直接包含影片
            video_files = [f for f in os.listdir(category_dir) if f.endswith((".mov", ".avi", ".mp4"))]
            if video_files:
                for video_file in video_files:
                    video_path = os.path.join(category_dir, video_file)
                    self.process_single_video(video_path, category)
            else:
                # 若有子資料夾，處理子資料夾中的影片
                for person_folder in os.listdir(category_dir):
                    person_folder_path = os.path.join(category_dir, person_folder)
                    if not os.path.isdir(person_folder_path):
                        continue

                    for video_file in os.listdir(person_folder_path):
                        if video_file.endswith((".mov", ".avi", ".mp4")):
                            video_path = os.path.join(person_folder_path, video_file)
                            self.process_single_video(video_path, category)

# 主程式
if __name__ == "__main__":
    print("Starting Video ROI Processor...")
    input_dir = "HDDM_Project/video"  # 輸入視頻目錄
    output_dir = "HDDM_Project/data_video"  # 輸出處理後的目錄

    processor = VideoROIProcessor(input_dir, output_dir)
    processor.process_videos()
