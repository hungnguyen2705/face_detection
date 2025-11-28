import cv2
import os
import sys
import time


SAVE_DIR = 'data_guong_mat'
MAX_IMAGES = 10  
FRAME_DELAY = 0.2

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


if len(sys.argv) < 3:
    print("Lỗi: Cần truyền Tên và Mã Sinh Viên.")
    sys.exit(1)

name = sys.argv[1] 
msv = sys.argv[2]  


safe_name = name.replace(" ", "_")
unique_id = f"{safe_name}_{msv}"

cap = cv2.VideoCapture(0)

print(f"--- CHẾ ĐỘ CHỤP ẢNH CHO: {name} ({msv}) ---")

count = 0
is_capturing = False
last_capture_time = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    
    display_frame = frame.copy()
    h, w, _ = display_frame.shape
    center_x, center_y = w // 2, h // 2

    if not is_capturing:
        cv2.rectangle(display_frame, (center_x - 120, center_y - 150), (center_x + 120, center_y + 150), (0, 255, 0), 2)
        cv2.putText(display_frame, "Nhan 's' de chup", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display_frame, f"SV: {name}", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        current_time = time.time()
        cv2.rectangle(display_frame, (center_x - 120, center_y - 150), (center_x + 120, center_y + 150), (0, 0, 255), 3)
        cv2.putText(display_frame, f"Dang chup: {count}/{MAX_IMAGES}", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if current_time - last_capture_time > FRAME_DELAY:
            
            filename = os.path.join(SAVE_DIR, f"{unique_id}_{count}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Lưu: {filename}")
            count += 1
            last_capture_time = current_time

        if count >= MAX_IMAGES:
            print("--> Đã chụp xong!")
            break

    cv2.imshow("Them Sinh Vien", display_frame)
    
    key = cv2.waitKey(1)
    if key == ord('s') and not is_capturing:
        is_capturing = True
        last_capture_time = time.time()
    elif key == ord('q'):
        sys.exit(1)

cap.release()
cv2.destroyAllWindows()
sys.exit(0)