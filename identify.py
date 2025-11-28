import cv2
import face_recognition
import os
import numpy as np
import pickle
import time
import sys
from ultralytics import YOLO


def get_base_path():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

BASE_PATH = get_base_path()

YOLO_MODEL = os.path.join(BASE_PATH, r'runs/detect/train4/weights/best.pt') 
DATABASE_FILE = os.path.join(BASE_PATH, 'face_database.pkl')

TOLERANCE = 0.45
RECHECK_INTERVAL = 2.0


print("1. Đang nạp Database...")
try:
    with open(DATABASE_FILE, 'rb') as f:
        database = pickle.load(f)
    known_encodings = list(database.values())
    known_names = list(database.keys())
    print(f"--> Đã load {len(known_names)} người.")
except FileNotFoundError:
    print("Lỗi: Chưa có database!")
    sys.exit()


print("2. Đang load YOLOv8 (Tracking Mode)...")
model = YOLO(YOLO_MODEL)

track_history = {} 

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    
    results = model.track(frame, persist=True, conf=0.5, verbose=False, tracker="bytetrack.yaml")
    
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
    else:
        boxes = []
        track_ids = []


    current_time = time.time()
    
    for box, track_id in zip(boxes, track_ids):
        x1, y1, x2, y2 = box
        
        name = "Unknown"
        color = (0, 0, 255) 
        
        
        should_recognize = False
        
        if track_id not in track_history:
            should_recognize = True 
        elif current_time - track_history[track_id]['last_check'] > RECHECK_INTERVAL:
            should_recognize = True 
        else:
            name = track_history[track_id]['name']

        
        if should_recognize:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_loc = [(y1, x2, y2, x1)] 
            
            face_encodings = face_recognition.face_encodings(rgb_frame, face_loc, model='large')
            
            found_name = "Unknown"
            if len(face_encodings) > 0:
                encoding = face_encodings[0]
                distances = face_recognition.face_distance(known_encodings, encoding)
                best_idx = np.argmin(distances)
                
                if distances[best_idx] < TOLERANCE:
                    found_name = known_names[best_idx]
            
            
            track_history[track_id] = {'name': found_name, 'last_check': current_time}
            name = found_name

        if name != "Unknown":
            color = (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(frame, (x1, y1 - 25), (x2, y1), color, cv2.FILLED)
        cv2.putText(frame, f"ID:{track_id} {name}", (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Tracking & Identification (Optimized)', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()