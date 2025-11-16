import cv2
import face_recognition
import os
import numpy as np
from ultralytics import YOLO

YOLO_MODEL = r'runs/detect/train4/weights/best.pt' 
FACE_DB_DIR = 'data_guong_mat'
TOLERANCE = 0.4

print("1. Đang nạp và xử lý dữ liệu khuôn mặt mẫu...")
known_encodings = []
known_names = []

temp_database = {}

if not os.path.exists(FACE_DB_DIR):
    print(f"error: {FACE_DB_DIR}")
    exit()

for file in os.listdir(FACE_DB_DIR):
    if file.endswith(('.jpg', '.png', '.jpeg')):
        path = os.path.join(FACE_DB_DIR, file)
        
        raw_name = os.path.splitext(file)[0]
        real_name = raw_name.split('_')[0]
        
        try:
            img = face_recognition.load_image_file(path)
            
            encs = face_recognition.face_encodings(img, model='large')
            
            if len(encs) > 0:
                if real_name not in temp_database:
                    temp_database[real_name] = []
                
                temp_database[real_name].append(encs[0])
        except Exception as e:
            print(f"Lỗi khi đọc file {file}: {e}")

print(" average vector ")
for name, encodings_list in temp_database.items():
    if len(encodings_list) > 0:
        avg_encoding = np.mean(encodings_list, axis=0)
        
        known_encodings.append(avg_encoding)
        known_names.append(name)
        print(f" + Đã tạo dữ liệu chuẩn cho: {name} (Dựa trên {len(encodings_list)} ảnh)")

print(f"done! {len(known_names)} .")

print("2. load YOLOv8...")
model = YOLO(YOLO_MODEL)

print("3. Khởi động Camera...")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    results = model.predict(frame, conf=0.5, verbose=False, imgsz=640)
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

    face_locations = []
    for box in boxes:
        x1, y1, x2, y2 = box
        face_locations.append((y1, x2, y2, x1))

    if len(face_locations) > 0:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, model='large')

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            
            name = "Unknown" 
            color = (0, 0, 255) 

            if len(distances) > 0:
                best_match_index = np.argmin(distances)
                distance_value = distances[best_match_index]

                if distance_value < TOLERANCE:
                    name = known_names[best_match_index]
                    color = (0, 255, 0) 
                    

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, top - 30), (right, top), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 5, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('He thong nhan dien khuon mat', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()