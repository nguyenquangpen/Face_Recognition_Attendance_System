from flask import Flask, request, jsonify
import os
import cv2
import mediapipe as mp
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle

app = Flask(__name__)

# Thư mục lưu video upload tạm thời
UPLOAD_FOLDER = "D:/Work/Face_Recognition_Attendance_System/Main/Model/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Thư mục lưu dataset khuôn mặt
DATASET_FOLDER = "D:/Work/Face_Recognition_Attendance_System/Main/Model/dataset"
os.makedirs(DATASET_FOLDER, exist_ok=True)

# Thư mục lưu model
MODEL_DIR = "D:/Work/Face_Recognition_Attendance_System/Main/Model/KNN_Model"
os.makedirs(MODEL_DIR, exist_ok=True)

@app.route('/process_video', methods=['POST'])
def process_video():
    name = request.form.get("name")
    video = request.files.get("video")

    if not name or not video:
        return jsonify({"status": "error", "message": "Thiếu tên hoặc video"}), 400

    save_dir = os.path.join(DATASET_FOLDER, name)
    os.makedirs(save_dir, exist_ok=True)

    video_path = os.path.join(UPLOAD_FOLDER, f"{name}.mp4")
    video.save(video_path)

    mp_face_detection = mp.solutions.face_detection
    saved = 0

    cap = cv2.VideoCapture(video_path)
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x = max(0, int(bbox.xmin * w))
                    y = max(0, int(bbox.ymin * h))
                    x2 = min(w, x + int(bbox.width * w))
                    y2 = min(h, y + int(bbox.height * h))
                    face = frame[y:y2, x:x2]

                    if face.size > 0:
                        cv2.imwrite(os.path.join(save_dir, f"{saved}.jpg"), face)
                        saved += 1

    cap.release()

    return jsonify({
        "status": "success",
        "saved": saved,
        "name": name
    })


def train_model_function():
    img_size = 150
    x = []
    y = []

    for person_name in os.listdir(DATASET_FOLDER):
        person_path = os.path.join(DATASET_FOLDER, person_name)
        if not os.path.isdir(person_path):
            continue
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)

            img = cv2.imread(img_path)
            img = cv2.resize(img, (img_size, img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_flatten = img.flatten() / 250.0
            x.append(img_flatten)
            y.append(person_name)

    x = np.array(x)
    y = np.array(y)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x, y)

    model_path = os.path.join(MODEL_DIR, "face_recognition_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(knn, f)

    print("✅ Đã huấn luyện và lưu model tại:", model_path)


@app.route('/train_model', methods=['POST'])
def train_model():
    try:
        print("🚀 Bắt đầu huấn luyện mô hình...")
        train_model_function()
        print("✅ Huấn luyện thành công")
        return jsonify({"status": "done"})
    except Exception as e:
        print("❌ Lỗi khi huấn luyện:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
