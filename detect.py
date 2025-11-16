from ultralytics import YOLO
import cv2

model_path = r'C:\Users\Admin\face_detection\runs\detect\train4\weights\best.pt'
model = YOLO(model_path)

cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("Không thể mở Webcam")
    exit()

print("Đang chạy nhận diện... Nhấn 'q' để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.5, verbose=False)
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv8 Face Detection - RTX 5070", annotated_frame)

    # Nhấn phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()