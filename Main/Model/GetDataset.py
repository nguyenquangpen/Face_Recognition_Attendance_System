import os
import cv2
import time
import mediapipe as mp

# Khởi tạo Mediapipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Nhập tên người dùng
name = input("Nhập tên người mới: ").strip()
save_dir = f"dataset/{name}"

# Kiểm tra nếu đã có người này
if os.path.exists(save_dir):
    print(f"Người dùng '{name}' đã tồn tại trong dataset.")
else:
    os.makedirs(save_dir, exist_ok=True)

    # Thời gian quay webcam
    capture_duration = 10
    start_time = time.time()
    saved = 0

    # Mở webcam
    cap = cv2.VideoCapture(0)
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x = max(0, int(bboxC.xmin * iw))
                    y = max(0, int(bboxC.ymin * ih))
                    w = int(bboxC.width * iw)
                    h = int(bboxC.height * ih)
                    x2 = min(iw, x + w)
                    y2 = min(ih, y + h)

                    # Cắt ảnh khuôn mặt
                    face_img = frame[y:y2, x:x2]
                    if face_img.size > 0:
                        cv2.imwrite(f"{save_dir}/{saved}.jpg", face_img)
                        saved += 1

                    mp_drawing.draw_detection(frame, detection)

            cv2.imshow("Dang ky khuon mat", frame)

            if time.time() - start_time > capture_duration:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Đã lưu {saved} ảnh vào: {save_dir}")
