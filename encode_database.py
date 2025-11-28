import face_recognition
import os
import numpy as np
import pickle


IMAGE_DIR = 'data_guong_mat'
DATABASE_FILE = 'face_database.pkl'

print("--- BẮT ĐẦU ENCODE DATABASE (TEN + MSV) ---")

temp_db = {}

if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

print(f"Đang quét thư mục: {IMAGE_DIR}...")
for file in os.listdir(IMAGE_DIR):
    if file.endswith(('.jpg', '.png', '.jpeg')):
        path = os.path.join(IMAGE_DIR, file)
        filename_no_ext = os.path.splitext(file)[0]
        
        parts = filename_no_ext.split('_')
        unique_id = "_".join(parts[:-1]) 
        
        try:
            img = face_recognition.load_image_file(path)
            encs = face_recognition.face_encodings(img, model='large')
            
            if len(encs) > 0:
                if unique_id not in temp_db:
                    temp_db[unique_id] = []
                temp_db[unique_id].append(encs[0])
                print(f" - Đã đọc: {file}")
        except Exception as e:
            print(f"Lỗi file {file}: {e}")

final_database = {}
for uid, vector_list in temp_db.items():
    if vector_list:
        avg_vector = np.mean(vector_list, axis=0)
        final_database[uid] = avg_vector

with open(DATABASE_FILE, 'wb') as f:
    pickle.dump(final_database, f)

print(f"\n--- HOÀN TẤT! Đã lưu {len(final_database)} sinh viên vào Database. ---")