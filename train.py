from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')
    results = model.train(
        data='C:/Users/Admin/face_detection/face-detection-4/data.yaml', 
        epochs=30, 
        imgsz=640,
        device=0  
    )