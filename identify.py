import cv2
import face_recognition
import os
import numpy as np
from ultralytics import YOLO
import pickle 


YOLO_MODEL = r'runs/detect/train4/weights/best.pt' 
DATABASE_FILE = 'face_database.pkl' 
TOLERANCE = 0.45

print("1. Đang nạp Database (Pre-computed)...")


try:
    with open(DATABASE_FILE, 'rb') as f:
        database = pickle.load(f)
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file {DATABASE_FILE}")
    print("Vui lòng chạy script 'encode_database.py' trước!")
    exit()


known_encodings = list(database.values()) 
known_names = list(database.keys())     

print(f"--> Sẵn sàng! Đã load {len(known_names)} người.")


print("2. Đang load YOLOv8...")
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
            
            name = "Nguoi La" 
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

    cv2.imshow('He thong nhan dien khuon mat (Pre-computed)', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()