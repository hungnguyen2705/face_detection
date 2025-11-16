import cv2
import os
import sys

SAVE_DIR = 'data_guong_mat'
IMAGES_PER_BATCH = 10
FRAME_SKIP = 4

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


if len(sys.argv) < 2:
    print("Lỗi: Cần truyền tên vào làm tham số.")
    sys.exit(1) 
    
name = sys.argv[1]
print(f"--- BẮT ĐẦU CHỤP ẢNH CHO: {name} ---")

PHASES = [
    {"id": "Truc_Dien", "msg": "Buoc 1: Nhin thang vao Camera"},
    {"id": "Ben_Phai",  "msg": "Buoc 2: Quay mat sang PHAI"},
    {"id": "Ben_Trai",  "msg": "Buoc 3: Quay mat sang TRAI"},
    {"id": "Nhin_Len",  "msg": "Buoc 4: Nguoc dau LEN tren"},
    {"id": "Nhin_Xuong","msg": "Buoc 5: Cui dau XUONG duoi"}
]

cap = cv2.VideoCapture(0)

current_phase_idx = 0
photos_taken = 0
frame_count = 0
is_capturing = False
user_quit = False

while True:
    ret, frame = cap.read()
    if not ret: break
    
    display_frame = frame.copy()
    
    if current_phase_idx >= len(PHASES):
        cv2.putText(display_frame, "DA HOAN THANH! (Tu dong tat...)", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Thu thap du lieu", display_frame)
        cv2.waitKey(2000) 
        break 
    
    current_phase = PHASES[current_phase_idx]
    phase_name = current_phase["id"]
    instruction = current_phase["msg"]

    if not is_capturing:
        cv2.putText(display_frame, instruction, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(display_frame, ">>> Nhan phim 's' de chup <<<", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.rectangle(display_frame, (150, 100), (490, 380), (0, 255, 0), 2)
        
    else:
        frame_count += 1
        if frame_count % FRAME_SKIP == 0:
            filename = f"{name}_{phase_name}_{photos_taken}.jpg"
            save_path = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(save_path, frame)
            print(f"Lưu: {filename}")
            photos_taken += 1

        cv2.putText(display_frame, f"DANG CHUP: {photos_taken}/{IMAGES_PER_BATCH}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.putText(display_frame, f"Goc: {phase_name}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.rectangle(display_frame, (0,0), (640,480), (0,0,255), 10)

        if photos_taken >= IMAGES_PER_BATCH:
            is_capturing = False
            photos_taken = 0
            current_phase_idx += 1
            print(f"--- Xong bước {phase_name} ---")

    cv2.imshow("Thu thap du lieu", display_frame)

    key = cv2.waitKey(1)
    if key == ord('s') and not is_capturing:
        is_capturing = True
        frame_count = 0
    elif key == ord('q'):
        user_quit = True
        break

cap.release()
cv2.destroyAllWindows()


if user_quit:
    print("Người dùng đã hủy.")
    sys.exit(1)
else:
    print("Thu thập dữ liệu hoàn tất.")
    sys.exit(0)