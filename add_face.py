import cv2
import os
import sys


SAVE_DIR = 'data_guong_mat'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


if len(sys.argv) < 2:
    print("Lỗi: Cần truyền tên vào.")
    sys.exit(1)

name = sys.argv[1]
cap = cv2.VideoCapture(0)

print(f"--- CHẾ ĐỘ THÊM MẶT: {name} ---")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    display_frame = frame.copy()
    h, w, _ = display_frame.shape
    

    cv2.putText(display_frame, "HAY NHIN THANG VAO CAMERA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(display_frame, "Nhan 's' de LUU - Nhan 'q' de HUY", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    

    center_x, center_y = w // 2, h // 2
    cv2.rectangle(display_frame, (center_x - 120, center_y - 150), (center_x + 120, center_y + 150), (0, 255, 0), 2)
    
    cv2.imshow("Them khuon mat", display_frame)
    
    key = cv2.waitKey(1)
    if key == ord('s'):

        filename = os.path.join(SAVE_DIR, f"{name}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Đã lưu ảnh: {filename}")
        sys.exit(0) 
        
    elif key == ord('q'):
        print("Đã hủy.")
        sys.exit(1) 

cap.release()
cv2.destroyAllWindows()