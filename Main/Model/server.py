from flask import Flask, request, jsonify
import os
import cv2
import mediapipe as mp
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle
import base64
import io
from PIL import Image
from flask_cors import CORS
import pymysql

app = Flask(__name__)
CORS(app)

# -------------------- Các thư mục --------------------
UPLOAD_FOLDER = "D:/Work/Face_Recognition_Attendance_System/Main/Model/uploads"
DATASET_FOLDER = "D:/Work/Face_Recognition_Attendance_System/Main/Model/dataset"
MODEL_DIR = "D:/Work/Face_Recognition_Attendance_System/Main/Model/KNN_Model"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------- Tham số --------------------
IMG_SIZE = 150
mp_face_detection = mp.solutions.face_detection

# -------------------- Load model (nếu tồn tại) --------------------
model_path = os.path.join(MODEL_DIR, "face_recognition_model.pkl")
model = None
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

# -------------------- API 1: Upload video + trích xuất ảnh --------------------
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

# -------------------- Hàm huấn luyện model --------------------
def train_model_function():
    global model
    x, y = [], []

    for person_name in os.listdir(DATASET_FOLDER):
        person_path = os.path.join(DATASET_FOLDER, person_name)
        if not os.path.isdir(person_path):
            continue
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_flatten = img.flatten() / 255.0
            x.append(img_flatten)
            y.append(person_name)

    x = np.array(x)
    y = np.array(y)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x, y)

    with open(model_path, 'wb') as f:
        pickle.dump(knn, f)

    model = knn

# -------------------- API 2: Huấn luyện model --------------------
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

# -------------------- API 3: Dự đoán từ ảnh base64 --------------------
def predict_face_from_image(image_bytes):
    if model is None:
        return {"status": "fail", "message": "Model chưa được huấn luyện."}

    image = Image.open(io.BytesIO(image_bytes))
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)

        if not results.detections:
            return {"status": "fail", "message": "Không phát hiện khuôn mặt."}

        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        h, w, _ = frame.shape
        x = max(0, int(bbox.xmin * w))
        y = max(0, int(bbox.ymin * h))
        x2 = min(w, x + int(bbox.width * w))
        y2 = min(h, y + int(bbox.height * h))
        face = frame[y:y2, x:x2]

        try:
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face_resized = cv2.resize(face_gray, (IMG_SIZE, IMG_SIZE))
            face_flatten = face_resized.flatten().reshape(1, -1) / 255.0
            pred = model.predict(face_flatten)[0]

            parts = pred.split("_", 1)
            student_id = parts[0]
            full_name = parts[1] if len(parts) > 1 else ""

            return {
                "status": "success",
                "student_id": student_id,
                "full_name": full_name
            }

        except:
            return {"status": "fail", "message": "Lỗi xử lý khuôn mặt."}

# -------------------- API 4: Điểm danh --------------------
@app.route("/predict_face", methods=["POST"])
def diemdanh():
    try:
        data = request.get_json()
        image_data = data.get("image_data")

        if not image_data:
            return jsonify({"status": "fail", "message": "Thiếu dữ liệu ảnh."}), 400

        header, encoded = image_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)

        result = predict_face_from_image(image_bytes)
        return jsonify(result)

    except Exception as e:
        return jsonify({"status": "fail", "message": "Lỗi server: " + str(e)}), 500


# --- Cấu hình kết nối MySQL ---
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "3105",
    "database": "face_Attendance",
    "cursorclass": pymysql.cursors.DictCursor
}

def get_db_connection():
    return pymysql.connect(**db_config)

# -------------------- API lưu dữ liệu sinh viên --------------------
@app.route("/save_student", methods=["POST"])
def save_student():
    try:
        data = request.get_json()
        student_id = data.get("student_id")
        full_name = data.get("full_name")
        class_name = data.get("class_name")

        if not all([student_id, full_name, class_name]):
            return jsonify({"status": "fail", "message": "Thiếu dữ liệu sinh viên."}), 400

        conn = get_db_connection()
        with conn.cursor() as cursor:
            # Kiểm tra xem student_id đã tồn tại chưa
            cursor.execute("SELECT * FROM students WHERE student_id=%s", (student_id,))
            existing = cursor.fetchone()

            if existing:
                # Nếu tồn tại thì cập nhật
                cursor.execute("""
                    UPDATE students SET full_name=%s, class_name=%s WHERE student_id=%s
                """, (full_name, class_name, student_id))
            else:
                # Nếu chưa tồn tại thì insert mới
                cursor.execute("""
                    INSERT INTO students (student_id, full_name, class_name) VALUES (%s, %s, %s)
                """, (student_id, full_name, class_name))

            conn.commit()
        conn.close()

        return jsonify({"status": "success", "message": "Điểm Danh thành công."})

    except Exception as e:
        return jsonify({"status": "fail", "message": "Lỗi điểm danh: " + str(e)}), 500


# -------------------- Run app --------------------
if __name__ == '__main__':
    app.run(debug=True)
