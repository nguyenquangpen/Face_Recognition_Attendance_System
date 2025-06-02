from flask import Flask, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import pickle
import base64
import io
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model KNN đã huấn luyện
with open('Main/Model/KNN_Model/face_recognition_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Thông số
IMG_SIZE = 150

mp_face_detection = mp.solutions.face_detection

def predict_face_from_image(image_bytes):
    # Chuyển ảnh base64 sang numpy array
    image = Image.open(io.BytesIO(image_bytes))
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)

        if not results.detections or len(results.detections) == 0:
            return {"status": "fail", "message": "Không phát hiện khuôn mặt."}

        # Lấy khuôn mặt đầu tiên
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

            # Tách student_id và tên
            parts = pred.split("_", 1)
            student_id = parts[0]
            full_name = parts[1] if len(parts) > 1 else ""

            return {
                "status": "success",
                "student_id": student_id,
                "full_name": full_name
            }

        except Exception as e:
            return {"status": "fail", "message": "Lỗi xử lý khuôn mặt."}

@app.route("/predict_face", methods=["POST"])
def diemdanh():
    try:
        data = request.get_json()
        image_data = data.get("image_data")

        if not image_data:
            return jsonify({"status": "fail", "message": "Thiếu dữ liệu ảnh."}), 400

        # Loại bỏ header của base64
        header, encoded = image_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)

        result = predict_face_from_image(image_bytes)

        return jsonify(result)

    except Exception as e:
        return jsonify({"status": "fail", "message": "Lỗi server: " + str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
# Chạy ứng dụng Flask
