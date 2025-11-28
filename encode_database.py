import face_recognition
import os
import numpy as np
import pickle
import sys


IMAGE_DIR = 'data_guong_mat'
DATABASE_FILE = 'face_database.pkl'

def encode_incremental():


    current_db = {}
    if os.path.exists(DATABASE_FILE):
        try:
            with open(DATABASE_FILE, 'rb') as f:
                current_db = pickle.load(f)
            print(f"--> Đã load dữ liệu cũ: {len(current_db)} người.")
        except Exception as e:
            print(f"Lỗi đọc file cũ (sẽ tạo mới): {e}")
            current_db = {}
    else:
        print("--> Chưa có Database, sẽ tạo mới hoàn toàn.")


    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
        print("Thư mục ảnh trống!")
        return

    files_by_id = {}
    print("-> Đang quét thư mục ảnh...")
    
    for file in os.listdir(IMAGE_DIR):
        if file.endswith(('.jpg', '.png', '.jpeg')):

            filename_no_ext = os.path.splitext(file)[0]
            parts = filename_no_ext.split('_')
            
            if len(parts) > 1:
                unique_id = "_".join(parts[:-1]) 
                
                if unique_id not in files_by_id:
                    files_by_id[unique_id] = []
                
                path = os.path.join(IMAGE_DIR, file)
                files_by_id[unique_id].append(path)


    new_ids_to_process = []
    skipped_count = 0
    
    for uid in files_by_id:
        if uid in current_db:
            skipped_count += 1
        else:
            new_ids_to_process.append(uid)

    print(f"--> Tìm thấy: {len(files_by_id)} sinh viên trong thư mục ảnh.")
    print(f"--> Đã biết: {skipped_count} người (BỎ QUA).")
    print(f"--> Cần học mới: {len(new_ids_to_process)} người.")

    if len(new_ids_to_process) == 0:
        print("\n=== KHÔNG CÓ DỮ LIỆU MỚI CẦN CẬP NHẬT ===")
        return


    print("\nBẮT ĐẦU MÃ HÓA NGƯỜI MỚI ")
    
    count_success = 0
    
    for uid in new_ids_to_process:
        image_paths = files_by_id[uid]
        print(f" + Đang học: {uid} ({len(image_paths)} ảnh)...")
        
        vectors = []
        for path in image_paths:
            try:
                img = face_recognition.load_image_file(path)

                encs = face_recognition.face_encodings(img, model='large')
                
                if len(encs) > 0:
                    vectors.append(encs[0])
            except Exception as e:
                print(f"   ! Lỗi ảnh {path}: {e}")

        if len(vectors) > 0:

            avg_vector = np.mean(vectors, axis=0)
            current_db[uid] = avg_vector
            count_success += 1
        else:
            print(f"   ! Cảnh báo: Không tìm thấy khuôn mặt nào của {uid}")


    if count_success > 0:
        with open(DATABASE_FILE, 'wb') as f:
            pickle.dump(current_db, f)
        print(f"\n=== CẬP NHẬT THÀNH CÔNG: Đã thêm {count_success} người mới ===")
        print(f"=== TỔNG CỘNG DATABASE: {len(current_db)} người ===")
    else:
        print("\n=== KHÔNG CÓ DỮ LIỆU HỢP LỆ ĐỂ LƯU ===")

if __name__ == "__main__":
    encode_incremental()