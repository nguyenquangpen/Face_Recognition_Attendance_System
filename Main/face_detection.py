import cv2
import numpy as np
import mediapipe as mp
import pickle

# Load model đã huấn luyện
with open('Model/face_reconition_model.pkl', 'rb') as f:
    model = pickle.load(f)


# Thông số
IMG_SIZE = 150

# Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
# Webcam
cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Chuyển ảnh sang RGB (Mediapipe yêu cầu RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)

                # Cắt khuôn mặt
                face_img = frame[y:y+h, x:x+w]
                try:
                    face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                    face_resized = cv2.resize(face_gray, (IMG_SIZE, IMG_SIZE))
                    face_flatten = face_resized.flatten().reshape(1, -1) / 255.0

                    # Dự đoán tên người
                    pred = model.predict(face_flatten)
                    name = pred[0]

                    # Vẽ khung và tên người
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                except:
                    pass

        cv2.imshow("Face Recognition - Diem danh", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
